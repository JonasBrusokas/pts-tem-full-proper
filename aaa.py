# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

from pts import Trainer
from pts.model.time_grad import TimeGradEstimator

# %%
if __name__ == "__main__":
    from utilz import *
    import uuid
    import pandas as pd  # Save the experiment arguments as a CSV file

    # Setup output directory
    output_dir = FileUtils.create_dir(
        os.path.join(FileUtils.project_root_dir(), "output_pts")
    )
    experiment_uuid = str(uuid.uuid4())
    experiment_output_dir = FileUtils.create_dir(
        os.path.join(output_dir, experiment_uuid)
    )
    start_timestamp = DateUtils.now()

    print("Experiment started: ", start_timestamp)
    print("Experiment with UUID created: ", experiment_uuid)

    # %%
    NUM_WORKERS = 0
    TEST_MODE = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device():

        def mps_available():
            try:
                return torch.backends.mps.is_available()
            except:
                return False

        if mps_available():
            print("MPS unsupported, using the CPU instead!")
            # return torch.device("cpu")
            return torch.device("cpu")
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    device = get_device()

    print("Using device: ", device)

    # %%
    def plot(
        target,
        forecast,
        prediction_length,
        prediction_intervals=(50.0, 90.0),
        color="g",
        fname=None,
    ):
        label_prefix = ""
        rows = 4
        cols = 4
        fig, axs = plt.subplots(rows, cols, figsize=(24, 24))
        axx = axs.ravel()
        seq_len, target_dim = target.shape

        ps = [50.0] + [
            50.0 + f * c / 2.0 for c in prediction_intervals for f in [-1.0, +1.0]
        ]

        percentiles_sorted = sorted(set(ps))

        def alpha_for_percentile(p):
            return (p / 100.0) ** 0.3

        for dim in range(0, min(rows * cols, target_dim)):
            ax = axx[dim]

            target[-2 * prediction_length :][dim].plot(ax=ax)

            ps_data = [forecast.quantile(p / 100.0)[:, dim] for p in percentiles_sorted]
            i_p50 = len(percentiles_sorted) // 2

            p50_data = ps_data[i_p50]
            p50_series = pd.Series(data=p50_data, index=forecast.index)
            p50_series.plot(color=color, ls="-", label=f"{label_prefix}median", ax=ax)

            for i in range(len(percentiles_sorted) // 2):
                ptile = percentiles_sorted[i]
                alpha = alpha_for_percentile(ptile)
                ax.fill_between(
                    forecast.index,
                    ps_data[i],
                    ps_data[-i - 1],
                    facecolor=color,
                    alpha=alpha,
                    interpolate=True,
                )
                # Hack to create labels for the error intervals.
                # Doesn't actually plot anything, because we only pass a single data point
                pd.Series(data=p50_data[:1], index=forecast.index[:1]).plot(
                    color=color,
                    alpha=alpha,
                    linewidth=10,
                    label=f"{label_prefix}{100 - ptile * 2}%",
                    ax=ax,
                )

        legend = ["observations", "median prediction"] + [
            f"{k}% prediction interval" for k in prediction_intervals
        ][::-1]
        axx[0].legend(legend, loc="upper left")

        if fname is not None:
            plt.savefig(fname, bbox_inches="tight", pad_inches=0.05)

    from run_commons import *

    parser = ExperimentUtils.get_arg_parse_for_fm(is_compat_old_ebm=True)
    args = parser.parse_args()

    from data_provider.experiment_data import ExperimentData
    import pandas as pd

    data_flag_to_use_for_training = "train"

    experiment_data = ExperimentData.from_args(
        args, train_flag=data_flag_to_use_for_training
    )
    loader = experiment_data.train_loader

    full_data_df = experiment_data.train_data.df_raw
    # Set 'date' as index
    full_data_df["date"] = pd.to_datetime(full_data_df["date"])
    full_data_df = full_data_df.set_index("date")

    train_data_df = full_data_df.iloc[: len(experiment_data.train_data)]
    val_data_df = full_data_df.iloc[
        len(experiment_data.train_data) : len(experiment_data.train_data)
        + len(experiment_data.val_data)
    ]
    test_data_df = full_data_df.iloc[
        len(experiment_data.train_data) + len(experiment_data.val_data) :
    ]

    def parse_freq(args):
        if args.data_path == "national_illness.csv":
            return "7D"
        elif args.data_path == "ETTh1.csv":
            return "1H"
        elif args.data_path == "ETTh2.csv":
            return "1H"
        elif args.data_path == "weather.csv":
            return "1H"
        elif args.data_path == "exchange_rate.csv":
            return "1D"
        elif args.data_path == "weather.csv":
            return "10T"
        else:
            raise Exception(f"Unsupported frequency: {args.freq}")

    def to_gluonts_dataset(df, target_column="OT", override_freq=None):
        feature_columns = list(set(df.columns).difference({"OT"}))
        freq_to_use = override_freq if override_freq is not None else parse_freq(args)
        df_to_use = df.copy()
        if args.data_path in [
            "national_illness.csv",
            "exchange_rate.csv",
            "weather.csv",
        ]:
            df_to_use = df_to_use.resample(freq_to_use).mean()
        dataset = PandasDataset(
            df_to_use,
            target=target_column,
            freq=freq_to_use,
            feat_dynamic_real=feature_columns,
        )
        return dataset

    train_gluonts_dataset = to_gluonts_dataset(train_data_df)
    test_gluonts_dataset = to_gluonts_dataset(test_data_df)

    ##%
    # Dynamically calculating features
    train_loader = experiment_data.train_loader
    feature_count = None
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        f_dim = -1 if args.features == "MS" else 0
        # outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len :, f_dim:]
        feature_count = batch_x.shape[-1]
        break

    max_target_dim = 1
    max_target_dim = feature_count

    train_grouper = MultivariateGrouper(max_target_dim=max_target_dim)
    # num_test_dates = int(len(dataset.test) / len(dataset.train))
    test_grouper = MultivariateGrouper(num_test_dates=1, max_target_dim=max_target_dim)
    dataset_train = train_grouper(train_gluonts_dataset)
    dataset_test = test_grouper(test_gluonts_dataset)

    # %%
    # Original
    # input_size = 1484
    # epochs = 20
    # num_batches_per_epoch = 100
    # num_forecast_samples = 100

    # Modified - testing
    # input_size = 64
    # batch_size = 8
    batch_size = 64
    if TEST_MODE:
        print("Running in test mode!")
        epochs = 2
        num_batches_per_epoch = 100
        num_forecast_samples = 10
        prediction_limit = 10
        print(
            f"Settings overriden: epochs={epochs}, num_batches_per_epoch={num_batches_per_epoch}, num_forecast_samples={num_forecast_samples}, prediction_limit={prediction_limit}"
        )
    else:
        # Proper
        epochs = 20
        num_batches_per_epoch = len(train_data_df) // batch_size
        num_forecast_samples = 10
        prediction_limit = None

    # target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
    # prediction_length = dataset.metadata.prediction_length
    # context_length = dataset.metadata.prediction_length

    target_dim = 1  # << Aligned with TEM
    prediction_length = 48  # << Aligned with TEM
    context_length = 96  # << Aligned with TEM
    input_size = feature_count + 1
    if args.data_path in ["exchange_rate.csv", "weather.csv"]:
        print("Overriding input size for exchange_rate.csv and weather.csv")
        input_size = feature_count

    estimator = TimeGradEstimator(
        target_dim=target_dim,
        prediction_length=prediction_length,
        context_length=context_length,
        cell_type="GRU",
        input_size=input_size,
        freq="H",  # <<< Needs to be constant, since this defines the input of the dataset!
        loss_type="l2",
        scaling=True,
        diff_steps=100,
        beta_end=0.1,
        beta_schedule="linear",
        trainer=Trainer(
            device=device,
            epochs=epochs,
            learning_rate=1e-3,
            num_batches_per_epoch=num_batches_per_epoch,
            batch_size=batch_size,
        ),
        # Updates:
        num_parallel_samples=200,
    )

    # %%
    predictor = estimator.train(dataset_train, num_workers=NUM_WORKERS)
    print(f"Training complete!")

    # %%
    # Add prediction code
    def make_predictions(
        predictor, test_df, context_length, prediction_length, set_name
    ):
        """
        Make predictions using a rolling window approach over the test set
        """
        all_predictions = []
        all_targets = []

        # Convert test_df to numpy for easier slicing
        test_data = test_df.values

        from utilz import DateUtils

        # Rolling window prediction
        before_loop = DateUtils.now()
        for i in range(0, len(test_df) - context_length - prediction_length + 1):
            # Get context window
            context_window = test_df.iloc[i : i + context_length]

            # Create a temporary dataset for this window
            window_dataset = to_gluonts_dataset(context_window)
            window_dataset = test_grouper(window_dataset)

            # Make prediction
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=window_dataset,
                predictor=predictor,
                num_samples=num_forecast_samples,
            )

            # Convert iterator to list to access predictions
            forecasts = list(forecast_it)
            targets = list(ts_it)

            all_predictions.append(forecasts[0])
            all_targets.append(targets[0])

            if i % 10 == 0:
                print(
                    f"[{set}] Completed prediction {i}/{len(test_df) - context_length - prediction_length}. Took: {DateUtils.now() - before_loop}"
                )
            if (prediction_limit is not None and prediction_limit >= 0) and (
                i >= prediction_limit
            ):
                print(
                    f"[{set}] Prediction limit reached: {prediction_limit}. Took: {DateUtils.now() - before_loop}"
                )
                break

        return all_predictions, all_targets

    def collect_predictions_and_targets(predictions, targets, prediction_length):
        predictions_sample_list = []
        predictions_median_list = []
        predictions_mean_list = []
        target_list = []

        for idx, (prediction, target) in enumerate(zip(predictions, targets)):
            # Save the prediction
            predictions_median_list.append(prediction.median)
            predictions_mean_list.append(prediction.mean)
            predictions_sample_list.append(prediction.samples)

            # Save the target
            target_cut = target.to_numpy()[-prediction_length:]
            target_list.append(target_cut)

        # Stack and concatenate the predictions and targets
        predictions_dict = {
            "pred_median": np.stack(predictions_median_list),
            "pred_mean": np.stack(predictions_mean_list),
            "pred_samples": np.stack(predictions_sample_list),
            "targets": np.stack(target_list),
        }

        return predictions_dict

    # >>> Make predictions on test set
    test_predictions, test_targets = make_predictions(
        predictor,
        test_data_df,
        context_length=context_length,
        prediction_length=prediction_length,
        set_name="test",
    )
    test_result_dict = collect_predictions_and_targets(
        test_predictions, test_targets, prediction_length
    )

    evaluator = Evaluator(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])
    test_agg_metrics, test_item_metrics = evaluator(test_targets, test_predictions)
    print("Test Aggregate metrics:")
    print(test_agg_metrics)

    # >>> Make predictions on validation set
    val_predictions, val_targets = make_predictions(
        predictor,
        val_data_df,
        context_length=context_length,
        prediction_length=prediction_length,
        set_name="val",
    )
    val_result_dict = collect_predictions_and_targets(
        val_predictions, val_targets, prediction_length
    )
    val_agg_metrics, val_item_metrics = evaluator(val_targets, val_predictions)
    print("Validation Aggregate metrics:")
    print(val_agg_metrics)

    """
    What do I need to output? (for val, test)
    4. Script arguments!
    
    1. Aggregate metrics (per item)
    2. Predictions
    3. Targets
    5. Indexes! So we can retrieve the items afterwards, if needed
    """

    # %%
    finish_timestamp = DateUtils.now()

    # %%
    # Output 'args'
    args_df = pd.DataFrame(vars(args), index=[0])
    args_df.to_csv(
        os.path.join(experiment_output_dir, "experiment_args.csv"), index=False
    )

    internal_args = {
        "epochs": epochs,
        "num_batches_per_epoch": num_batches_per_epoch,
        "num_forecast_samples": num_forecast_samples,
        "prediction_limit": prediction_limit,
        "target_dim": target_dim,
        "prediction_length": prediction_length,
        "context_length": context_length,
        "input_size": input_size,
        "batch_size": batch_size,
        "TEST_MODE": TEST_MODE,
        "device": str(device),
    }
    internal_args_df = pd.DataFrame(internal_args, index=[0])
    internal_args_df.to_csv(
        os.path.join(experiment_output_dir, "experiment_internal_args.csv"), index=False
    )

    # %%
    # Export predictions and targets on validation and test sets
    val_result_file_path = os.path.join(experiment_output_dir, "val_results.npz")
    np.savez(
        val_result_file_path,
        **val_result_dict,
    )

    test_result_file_path = os.path.join(experiment_output_dir, "test_results.npz")
    np.savez(
        test_result_file_path,
        **test_result_dict,
    )

    """
    - We need to implement the output
    - We need to setup and run on the VMs
    """

    # %%
    # Loaded the saved results

    try:
        test_results_loaded = dict(np.load(test_result_file_path))
        val_results_loaded = dict(np.load(val_result_file_path))
    except Exception as e:
        print("Error loading the saved results!")

    # %%
    # Write a text file called 'done.txt' indicating that the experiment has outputted everything succcessfully!
    # It should contain the finish_timestamp
    with open(os.path.join(experiment_output_dir, "done.txt"), "w") as f:
        f.write(
            f"{start_timestamp}\n{finish_timestamp}\n{finish_timestamp-start_timestamp}"
        )
