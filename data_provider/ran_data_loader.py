from datetime import timedelta

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils.timefeatures import time_features
from utilz import *


class Dataset_RAN(Dataset):
    DEFAULT_LABELS_TO_USE = [
        "ddos-ripper-C",
        "youtube",
        "Web Browsing",
        "portscan",
        "dos-hulk-C",
        "SIPP",
        "iot",
        "slowloris-C",
    ]
    SMALL_LABELS_TO_USE = ["iot"]
    SMALL_LABELS_TO_USE_2 = ["youtube"]
    SMALL_LABELS_TO_USE_3 = ["ddos-ripper-C"]

    DEFAULT_USER_IDS_TO_USE = [1, 2, 3, 4]
    SMALL_USER_IDS_TO_USE = [1]

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="MS",
        data_path="ran.csv",
        target="mac_dl_brate",
        scale=True,
        timeenc=0,
        freq="h",
        user_ids_to_use=None,
        labels_to_use=None,
        override_scaler=None,  # Added this line
    ):
        if user_ids_to_use is None:
            user_ids_to_use = Dataset_RAN.DEFAULT_USER_IDS_TO_USE
        if labels_to_use is None:
            labels_to_use = Dataset_RAN.DEFAULT_LABELS_TO_USE
        if size is None:
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
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.user_ids_to_use = user_ids_to_use
        self.labels_to_use = labels_to_use

        # Added override_scaler to the instance variables
        self.override_scaler = override_scaler

        # For debugging
        self.df_raw = None
        self.df_stamp_orig = None

        self.__read_data__()

    @staticmethod
    def convert_ts_to_datetime(timestamp) -> datetime:
        # TODO: Dunno what the magic constant is
        magic_constant = 100_000
        seconds, microseconds = divmod(timestamp, magic_constant)
        # Convert to a datetime object
        dt = datetime(1970, 1, 1) + timedelta(
            seconds=seconds, microseconds=microseconds
        )
        return dt

    def __read_data__(self):
        # Initialize the scaler
        if self.override_scaler is not None:
            self.scaler = self.override_scaler
            print("Using overridden scaler for Dataset_RAN.")
        else:
            self.scaler = StandardScaler()
            print("Creating new scaler for Dataset_RAN.")

        # NOTE: we use ';' as a separator
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), sep=";")

        # Set additional time
        df_raw["date_from_timestamp"] = df_raw["timestamp"].apply(
            lambda ts_item: Dataset_RAN.convert_ts_to_datetime(ts_item)
        )
        df_raw = df_raw.set_index("timestamp", drop=True)
        df_raw = df_raw.sort_index(axis=0)
        len_before_filter = len(df_raw)

        # We filter according to user column
        df_raw = df_raw[df_raw["id_ue"].isin(self.user_ids_to_use)]
        df_raw = df_raw[df_raw["label"].isin(self.labels_to_use)]

        print(
            f"Length of RAN dataset: {len(df_raw)}/{len_before_filter} for user ids: {self.user_ids_to_use}"
        )

        cols = [
            "mac_dl_cqi",
            "mac_dl_mcs",
            "mac_dl_ok",
            "mac_dl_nok",
            "mac_dl_buffer",
            "mac_dl_brate",
        ]
        df_raw = df_raw[cols]

        # Filter with specific columns immediately
        self.df_raw = df_raw

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[:-1]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Features {self.features} not supported")

        if self.scale:
            if self.override_scaler is None:
                # Only fit the scaler if override_scaler is not provided
                train_data = df_data.iloc[border1s[0] : border2s[0]]
                self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        def convert_ts_to_datetime(timestamp) -> datetime:
            magic_constant = 1_000_000
            seconds, microseconds = divmod(timestamp, magic_constant)
            # Convert to a datetime object
            dt = datetime(1970, 1, 1) + timedelta(
                seconds=seconds, microseconds=microseconds
            )
            return dt

        df_stamp = df_raw.index[border1:border2]
        df_stamp = pd.DataFrame({"date": df_stamp})
        df_stamp["date"] = df_stamp.apply(
            lambda x: convert_ts_to_datetime(x["date"]), axis=1
        )
        self.df_stamp_orig = df_stamp
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train_data(self):
        df_raw = self.df_raw
        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"self.features === '{self.features}' unsupported!")

        train_data = df_data.iloc[border1s[0] : border2s[0]]
        return train_data


if __name__ == "__main__":
    root_path = os.path.join(FileUtils.project_root_dir(), "dataset")
    data_path = "ran.csv"
    flag = "train"
    seq_len = 24
    label_len = 20
    pred_len = 4
    features = "MS"  # TODO: try MS and M
    target = "mac_dl_bitrate"
    timeenc = 1
    # freq = "h"  # TODO: make 15 min

    dataset = Dataset_RAN(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=timeenc,
        # freq=freq, # TODO: 'freq' unclear
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
    #

    df_raw = dataset.df_raw
    ts_df = pd.to_datetime(df_raw.index)
