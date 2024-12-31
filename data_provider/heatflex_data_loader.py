import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, ConcatDataset

from data_provider.data_loader import Dataset_Custom
from utils.timefeatures import time_features
from utilz import *


class HeatPumpDatasetNYC:
    # Col definitions
    NEW_DATETIME_COL = "DateTime"
    DATETIME_COL = NEW_DATETIME_COL
    AMBIENT_TEMP_COL = "Ambient Outdoor Temperature"
    ROOM_TEMP_COL = "Room Air Temperature"
    POWER_COL = "Total Unit Power"
    SITE_ID_COL = "Site Identification"
    DATE_COL = "Date"
    TIME_COL = "Time"
    CONTROL_SIGNAL_COL = "Control Signal"  # Generated

    INDOOR_TEMP_COL = ROOM_TEMP_COL
    OUTDOOR_TEMP_COL = AMBIENT_TEMP_COL


class DataPreparationUtils:
    DEFAULT_OUTLIER_TEMP_THRESHOLD: float = 15.0

    @classmethod
    # Warning: outlier_fix modifies the 'df' instead of creating a copy
    def outlier_fix(
        cls,
        orig_df: pd.DataFrame,
        room_temp_col: str = HeatPumpDatasetNYC.INDOOR_TEMP_COL,
        ambient_temp_col: str = HeatPumpDatasetNYC.OUTDOOR_TEMP_COL,
        outlier_temp_threshold: float = DEFAULT_OUTLIER_TEMP_THRESHOLD,
    ):
        df = orig_df.copy()

        i = 1
        length_df = len(df)
        prev_proper_room_temp: Optional[float] = None
        prev_proper_ambient_temp: Optional[float] = None

        while i < length_df:
            prev_record = df.iloc[i - 1]
            current_record = df.iloc[i]

            # Dealing with ambient temperature
            if prev_proper_room_temp is not None:
                if (
                    abs(prev_proper_room_temp - current_record[room_temp_col])
                    <= outlier_temp_threshold
                ):
                    prev_proper_room_temp = None
                else:
                    df.iloc[i][room_temp_col] = np.nan
            elif (
                abs(prev_record[room_temp_col] - current_record[room_temp_col])
                > outlier_temp_threshold
            ):
                # print(f"PROBLEM room_temp at i={i}, prev={prev_record}, curr={current_record}")
                prev_proper_room_temp = prev_record[room_temp_col]
                df.iloc[i][room_temp_col] = np.nan

            # Dealing with ambient temperature
            if prev_proper_ambient_temp is not None:
                if (
                    abs(prev_proper_ambient_temp - current_record[ambient_temp_col])
                    <= outlier_temp_threshold
                ):
                    prev_proper_ambient_temp = None
                else:
                    df.iloc[i][ambient_temp_col] = np.nan
            elif (
                abs(prev_record[ambient_temp_col] - current_record[ambient_temp_col])
                > outlier_temp_threshold
            ):
                # print(f"PROBLEM ambient_temp at i={i}, prev={prev_record}, curr={current_record}")
                prev_proper_ambient_temp = prev_record[ambient_temp_col]
                df.iloc[i][ambient_temp_col] = np.nan
            i += 1

        return df

    @classmethod
    def site_subset_df(cls, df: pd.DataFrame, id: str) -> pd.DataFrame:
        # Method for selecting a specific site
        return df[df[HeatPumpDatasetNYC.SITE_ID_COL] == id].drop(
            columns=[HeatPumpDatasetNYC.SITE_ID_COL]
        )


def check_if_heatflex_dataset(data: Dataset_Custom):
    return isinstance(data, Dataset_Heatflex) or isinstance(data, Dataset_HeatflexMulti)


class Dataset_Heatflex(Dataset_Custom):
    def __init__(
        self,
        root_path,
        site_id,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target=HeatPumpDatasetNYC.INDOOR_TEMP_COL,
        scale=True,
        timeenc=0,
        freq="h",
        override_scaler=None,
    ):
        self.site_id = site_id
        self.override_scaler = override_scaler
        self.train_data = None
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq
        )

    def __read_data__(self):
        if self.target != HeatPumpDatasetNYC.INDOOR_TEMP_COL:
            print(
                f"Y target overriden: from '{self.target}' to '{HeatPumpDatasetNYC.INDOOR_TEMP_COL}'"
            )
            self.target = HeatPumpDatasetNYC.INDOOR_TEMP_COL

        self.scaler = StandardScaler()

        intermediate_heatflex_data_df_list = list(
            map(
                lambda data_path: pd.read_csv(
                    os.path.join(self.root_path, data_path),
                    index_col=HeatPumpDatasetNYC.NEW_DATETIME_COL,
                    parse_dates=True,
                ),
                ["NIST_data_15min.csv", "NYC_data_15min.csv"],
            )
        )

        # heatflex_data_df = pd.read_csv(
        #     os.path.join(self.root_path, self.data_path),
        #     index_col=HeatPumpDatasetNYC.NEW_DATETIME_COL,
        #     parse_dates=True,
        # )
        heatflex_data_df = pd.concat(intermediate_heatflex_data_df_list)

        # site_df = heatflex_data_df[heatflex_data_df[HeatPumpDatasetNYC.SITE_ID_COL] == 'S44']
        site_df = DataPreparationUtils.site_subset_df(heatflex_data_df, self.site_id)
        site_df = site_df[
            [
                HeatPumpDatasetNYC.INDOOR_TEMP_COL,
                HeatPumpDatasetNYC.OUTDOOR_TEMP_COL,
                HeatPumpDatasetNYC.POWER_COL,
            ]
        ]
        site_df = DataPreparationUtils.outlier_fix(site_df, outlier_temp_threshold=15.0)
        site_df = site_df.resample("15min").asfreq()

        site_df[HeatPumpDatasetNYC.INDOOR_TEMP_COL] = site_df[
            HeatPumpDatasetNYC.INDOOR_TEMP_COL
        ].interpolate()
        site_df[HeatPumpDatasetNYC.OUTDOOR_TEMP_COL] = site_df[
            HeatPumpDatasetNYC.OUTDOOR_TEMP_COL
        ].interpolate()

        # NOTE 2022-11-14 - Make sure that filling with 0s is verified. In other pipeline items were dropped instead
        site_df[HeatPumpDatasetNYC.POWER_COL] = site_df[
            HeatPumpDatasetNYC.POWER_COL
        ].fillna(value=0.0)
        self.power_on_threshold = site_df[HeatPumpDatasetNYC.POWER_COL].quantile(0.95)

        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(f">>> {self.site_id} ON threshold: {self.power_on_threshold} ")
        # quantiles = [0.85, 0.9, 0.95, 0.99]
        # print(
        #     quantiles,
        #     "\n",
        #     [site_df[HeatPumpDatasetNYC.POWER_COL].quantile(q) for q in quantiles],
        # )
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        site_df = site_df.reset_index()

        # Adding control signal U
        site_df[HeatPumpDatasetNYC.CONTROL_SIGNAL_COL] = site_df[
            HeatPumpDatasetNYC.POWER_COL
        ].shift(
            periods=-self.pred_len,  # Number of periods to shift
        )
        site_df = site_df.head(len(site_df) - self.pred_len)

        self.df_raw = df_raw = site_df
        date_col = HeatPumpDatasetNYC.NEW_DATETIME_COL

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove(date_col)

        # Reordering the columns
        self.reordered_columns = [date_col] + cols + [self.target]
        df_raw = df_raw[self.reordered_columns]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]  # 0 - train, 1 - val, 2 - test (type_map)
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        # We save it for the multi dataset case
        self.train_data = df_data[border1s[0] : border2s[0]]
        if self.scale:
            if self.override_scaler is None:
                self.scaler.fit(self.train_data.values)
            else:
                # print(" <<< USING THE OVERRIDE SCALER >>> ")
                self.scaler = self.override_scaler
                # print(f"Scaler (internal): {self.scaler}")
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            self.train_data = None

        df_stamp = df_raw[[date_col]][border1:border2]
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([date_col], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp[date_col].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return super().__len__()

    def inverse_transform(self, data):
        return super().inverse_transform(data)

    # Redundant, but lets give it a go
    def get_train_data(self):
        date_col = HeatPumpDatasetNYC.NEW_DATETIME_COL

        cols = list(self.df_raw.columns)
        cols.remove(self.target)
        cols.remove(date_col)

        # Reordering the columns
        self.reordered_columns = [date_col] + cols + [self.target]
        df_raw = self.df_raw[self.reordered_columns]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        # border1 = border1s[self.set_type]  # 0 - train, 1 - val, 2 - test (type_map)
        # border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"self.features === '{self.features}' unsupported!")

        # We save it for the multi dataset case
        train_data = df_data[border1s[0] : border2s[0]]
        return train_data


class Dataset_HeatflexMulti(Dataset_Custom):
    scaler_cache = {}
    dataset_cache = {}

    @staticmethod
    def add_scaler_to_cache(key_joined, scaler_to_cache):
        Dataset_HeatflexMulti.scaler_cache[key_joined] = scaler_to_cache

    @staticmethod
    def fetch_scaler_from_cache(key_joined):
        return Dataset_HeatflexMulti.scaler_cache[key_joined]

    @staticmethod
    def fetch_dataset_from_cache(key_joined):
        (
            concat_dataset_from_cache,
            site_datasets_from_cache,
        ) = Dataset_HeatflexMulti.dataset_cache[key_joined]
        return concat_dataset_from_cache, site_datasets_from_cache

    @staticmethod
    def add_dataset_to_cache(
        key_joined, concat_dataset_to_cache, site_datasets_to_cache
    ):
        Dataset_HeatflexMulti.dataset_cache[key_joined] = (
            concat_dataset_to_cache,
            site_datasets_to_cache,
        )

    def __init__(
        self,
        root_path: str,
        site_id_list: [str],
        flag: str = "train",
        size=None,
        features="MS",
        data_path="NYC_data_15min.csv",
        target=HeatPumpDatasetNYC.INDOOR_TEMP_COL,
        scale=True,
        timeenc=0,
        freq="t",  # TODO: check if "15min" is properly supported!
        override_scaler=None,
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq
        )
        self.override_scaler = override_scaler

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        if self.target != HeatPumpDatasetNYC.INDOOR_TEMP_COL:
            print(
                f"Y target overriden: from '{self.target}' to '{HeatPumpDatasetNYC.INDOOR_TEMP_COL}'"
            )
            self.target = HeatPumpDatasetNYC.INDOOR_TEMP_COL

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.site_id_list = site_id_list

        self.root_path = root_path
        self.data_path = data_path

        def setup_single_site_dataset(site_id, fitted_scaler=None):
            return Dataset_Heatflex(
                root_path,
                site_id,
                flag,
                size,
                features,
                data_path,
                self.target,
                fitted_scaler
                is not None,  # NOTE: we are not scaling inner ones, we have to handle explicitly
                timeenc,
                freq,
                override_scaler=fitted_scaler,
            )

        site_id_key_joined = ",".join(site_id_list)

        if site_id_key_joined not in Dataset_HeatflexMulti.scaler_cache:
            # If the scaler is not cached - compute it
            print(
                f"Scaler for multi-site HeatFlex ({site_id_key_joined}) dataset not found"
            )
            self.orig_site_datasets = list(
                map(lambda site_id: setup_single_site_dataset(site_id), site_id_list)
            )
            self.orig_concat_site_dataset = ConcatDataset(self.orig_site_datasets)
            self.orig_concat_site_dataset_train = pd.concat(
                list(
                    map(
                        lambda site_dataset: site_dataset.get_train_data(),
                        self.orig_site_datasets,
                    )
                )
            )

            self.concat_site_scaler = StandardScaler().fit(
                self.orig_concat_site_dataset_train
            )
            Dataset_HeatflexMulti.add_scaler_to_cache(
                site_id_key_joined, scaler_to_cache=self.concat_site_scaler
            )
        else:
            print(
                f"Scaler for multi-site HeatFlex ({site_id_key_joined}) dataset FOUND!"
            )
            self.concat_site_scaler = Dataset_HeatflexMulti.fetch_scaler_from_cache(
                site_id_key_joined
            )

        site_id_and_flag = f"{site_id_key_joined}_{flag}"
        if site_id_and_flag not in Dataset_HeatflexMulti.dataset_cache:
            print(
                f"Datasets for multi-site HeatFlex ({site_id_and_flag}) dataset not found!"
            )
            self.site_datasets = list(
                map(
                    lambda site_id: setup_single_site_dataset(
                        site_id, fitted_scaler=self.concat_site_scaler
                    ),
                    site_id_list,
                )
            )
            self.concat_site_dataset = ConcatDataset(self.site_datasets)
            self.add_dataset_to_cache(
                key_joined=site_id_and_flag,
                concat_dataset_to_cache=self.concat_site_dataset,
                site_datasets_to_cache=self.site_datasets,
            )
        else:
            print(
                f"Datasets for multi-site HeatFlex ({site_id_and_flag}) dataset FOUND!"
            )
            (
                self.concat_site_dataset,
                self.site_datasets,
            ) = self.fetch_dataset_from_cache(site_id_and_flag)

        # For compatibility!
        self.scaler = self.concat_site_scaler

    def __read_data__(self):
        # Just a NOP method
        pass

    def __getitem__(self, index):
        return self.concat_site_dataset.__getitem__(index)

    def __len__(self):
        return self.concat_site_dataset.__len__()

    def inverse_transform(self, data):
        # return super().inverse_transform(data)
        raise ValueError("NOT IMPLEMENTED")


if __name__ == "__main__":
    # Testing if this works

    root_path = os.path.join(FileUtils.project_root_dir(), "dataset")
    site_id = "S44"
    data_path = "NYC_data_15min.csv"
    flag = "train"
    seq_len = 24
    label_len = 20
    pred_len = 4
    features = "MS"  # TODO: try MS and M
    target = HeatPumpDatasetNYC.INDOOR_TEMP_COL
    timeenc = 1
    freq = "h"  # TODO: make 15 min

    example_dataset = Dataset_Custom(
        root_path=root_path,
        data_path="ETTh1.csv",
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target="OT",
        timeenc=timeenc,
        freq=freq,
    )

    dataset = Dataset_Heatflex(
        root_path=root_path,
        site_id=site_id,
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=timeenc,
        freq=freq,
    )

    batch_size = 16
    drop_last = True
    num_workers = 0
    shuffle_flag = True

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        batch_x, batch_y, batch_x_mark, batch_y_mark = (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
        )
        break

    # %%

    dataset_multi = Dataset_HeatflexMulti(
        root_path=root_path,
        site_id_list=["S40", "S44"],
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=timeenc,
        freq=freq,
    )

    train_loader_multi = DataLoader(
        dataset_multi,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
        train_loader_multi
    ):
        batch_x, batch_y, batch_x_mark, batch_y_mark = (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
        )
        break

    site_dataset = dataset_multi.site_datasets[0]
    site_dataset_train_data = site_dataset.train_data
