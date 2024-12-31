#%%
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.repository.datasets import get_dataset

# Lets query some of the datasets, shall we? :)

# TODO: manually import this...
# ett_dataset = get_dataset("ett_small_1h", regenerate=False)

from run_commons import *

parser = ExperimentUtils.get_arg_parse_for_fm(is_compat_old_ebm=True)
args = parser.parse_args()

#%%

# NOTE: needs to be in main for some reason
if __name__ == '__main__':
    # Works
    weather_dataset = get_dataset("weather", regenerate=True)
    #
    # #
    # exchange_rate_dataset = get_dataset("exchange_rate", regenerate=True)

    ##%%
    from data_provider.experiment_data import ExperimentData
    import pandas as pd

    data_flag_to_use_for_training = "train"

    experiment_data = ExperimentData.from_args(
        args, train_flag=data_flag_to_use_for_training
    )
    loader = experiment_data.train_loader

    full_data_df = experiment_data.train_data.df_raw
    train_data_df = full_data_df.iloc[:len(experiment_data.train_data)]
    val_data_df = full_data_df.iloc[len(experiment_data.train_data):len(experiment_data.train_data) + len(experiment_data.val_data)]
    test_data_df = full_data_df.iloc[len(experiment_data.train_data) + len(experiment_data.val_data):]

    def to_gluonts_dataset(df, target_column="OT"):
        dataset = PandasDataset(df, target=target_column, freq="1H")
        return dataset

    train_gluonts_dataset = to_gluonts_dataset(train_data_df)

    #%%

    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName

    def torch_to_gluonts(data_loader, args):
        gluonts_list = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                data_loader
        ):
            f_dim = -1 if args.features == "MS" else 0
            # outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            # Assuming item is a tuple of (features, target)
            gluonts_item = {
                FieldName.TARGET: batch_y.numpy(),
                FieldName.START: pd.Timestamp("2021-01-01"),  # Replace with actual start date
                # FieldName.FEAT_STATIC_CAT: features.numpy() if features is not None else None
                FieldName.FEAT_STATIC_CAT: None,
            }
            gluonts_list.append(gluonts_item)

        return ListDataset(gluonts_list, freq=args.freq)


    # Usage
    # gluonts_dataset = torch_to_gluonts(data_loader=loader, args=args)

