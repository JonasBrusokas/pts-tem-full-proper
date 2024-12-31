#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

from pts.model.time_grad import TimeGradEstimator
from pts import Trainer

#%%
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device():
    if torch.backends.mps.is_available():
        print("MPS unsupported, using the CPU instead")
        return torch.device("cpu")
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()

print("Using device: ", device)

# %%
def plot(target, forecast, prediction_length, prediction_intervals=(50.0, 90.0), color='g', fname=None):
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

        target[-2 * prediction_length:][dim].plot(ax=ax)

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

    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    axx[0].legend(legend, loc="upper left")

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.05)


# %%
print(f"Available datasets: {list(dataset_recipes.keys())}")
# %%
# exchange_rate_nips, electricity_nips, traffic_nips, solar_nips, wiki-rolling_nips, ## taxi_30min is buggy still
dataset = get_dataset("electricity_nips", regenerate=False)
# %%
dataset.metadata
# %%
train_grouper = MultivariateGrouper(max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))

num_test_dates = int(len(dataset.test) / len(dataset.train))
test_grouper = MultivariateGrouper(num_test_dates=num_test_dates,
                                   max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))
# %%
dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)
# %%
# Original
input_size = 1484
batch_size = 64
epochs = 20
num_batches_per_epoch = 100
num_forecast_samples = 100

# Modified
# input_size = 64
# batch_size = 8
epochs = 1
num_batches_per_epoch = 2
num_forecast_samples = 2

estimator = TimeGradEstimator(
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=dataset.metadata.prediction_length,
    context_length=dataset.metadata.prediction_length,
    cell_type='GRU',
    input_size=input_size,
    freq=dataset.metadata.freq,
    loss_type='l2',
    scaling=True,
    diff_steps=100,
    beta_end=0.1,
    beta_schedule="linear",
    trainer=Trainer(device=device,
                    epochs=epochs,
                    learning_rate=1e-3,
                    num_batches_per_epoch=num_batches_per_epoch,
                    batch_size=batch_size, ),
    # Updates:
    num_parallel_samples=1,
)

# for batch in dataset_train:
#     print("Input shape:", batch["past_target"].shape)
#     print("Feature shape:", batch["past_feat_dynamic_real"].shape if "past_feat_dynamic_real" in batch else "No features")
#     break
#
# raise ValueError("Stop here")

# %%
predictor = estimator.train(dataset_train, num_workers=NUM_WORKERS)
print(f"Training complete!")

# %%
forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                                 predictor=predictor,
                                                 num_samples=num_forecast_samples)

# %%
forecasts = list(forecast_it)
targets = list(ts_it)
# %%
plot(
    target=targets[0],
    forecast=forecasts[0],
    prediction_length=dataset.metadata.prediction_length,
)
plt.show()
# %%
evaluator = MultivariateEvaluator(quantiles=(np.arange(20) / 20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})
# %%
agg_metric, item_metrics = evaluator(targets, forecasts, num_series=len(dataset_test))
# %%
print("CRPS:", agg_metric["mean_wQuantileLoss"])
print("ND:", agg_metric["ND"])
print("NRMSE:", agg_metric["NRMSE"])
print("")
print("CRPS-Sum:", agg_metric["m_sum_mean_wQuantileLoss"])
print("ND-Sum:", agg_metric["m_sum_ND"])
print("NRMSE-Sum:", agg_metric["m_sum_NRMSE"])

# %%
# Lets try to hack together some sort of selective forecasting, shall we?

print("Number of forecasts made: ", len(forecasts))

a = forecasts[0]

#%%
###

# Lets query some of the datasets, shall we? :)

# TODO: manually import this...
# ett_dataset = get_dataset("ett_small_1h", regenerate=False)

# Works
# weather_dataset = get_dataset("weather", regenerate=False)
#
# #
# exchange_rate_dataset = get_dataset("exchange_rate", regenerate=False)

#%%
from data_provider.experiment_data import ExperimentData

args = ()
data_flag_to_use_for_training = "train"

experiment_data = ExperimentData.from_args(
    args, train_flag=data_flag_to_use_for_training
)

