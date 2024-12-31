from argparse import Namespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from data_provider.m4 import M4Meta, M4Dataset
from utilz import *


class Dataset_M4(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="pred",
        size=None,
        features="S",
        data_path="ETTh1.csv",  # ignored
        target="OT",
        scale=False,
        inverse=False,
        timeenc=0,
        freq="15min",  # ignored
        seasonal_patterns="Yearly",
    ):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        if self.seasonal_patterns not in M4Meta.seasonal_patterns:
            raise ValueError(
                f"Set M4 seasonal patterns '{self.seasonal_patterns}' are invalid! Available: {M4Meta.seasonal_patterns}"
            )
        if self.features != "S":
            raise ValueError(
                "Features 'S' is the only supported configuration for the M4 dataset"
            )
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag
        self.scaler = None

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == "train":
            dataset = M4Dataset.load(
                training=True,
            )
        elif self.flag == "val":
            print(f"TEMPORARILY using 'test' set as a validation set")
            dataset = M4Dataset.load(
                training=False,
            )
        else:
            dataset = M4Dataset.load(
                training=False,
            )
        training_values = np.asarray(
            [
                v[~np.isnan(v)]
                for v in dataset.values[dataset.groups == self.seasonal_patterns]
            ],
            dtype="object",
        )  # split different frequencies
        self.ids = np.asarray(
            [i for i in dataset.ids[dataset.groups == self.seasonal_patterns]],
            dtype="object",
        )
        self.timeseries = [ts for ts in training_values]

        # We duplicate everything!
        self.train_dataset = M4Dataset.load(
            training=True,
        )
        self.test_dataset = M4Dataset.load(
            training=False,
        )

        def prepare_timeseries_from_dataset(
            dataset, seasonal_patterns=self.seasonal_patterns
        ):
            training_values = np.asarray(
                [
                    v[~np.isnan(v)]
                    for v in dataset.values[dataset.groups == seasonal_patterns]
                ],
                dtype="object",
            )  # split different frequencies
            ids = np.asarray(
                [i for i in dataset.ids[dataset.groups == seasonal_patterns]],
                dtype="object",
            )
            timeseries = [ts for ts in training_values]
            return timeseries, ids

        train_ts, train_ids = prepare_timeseries_from_dataset(self.train_dataset)
        test_ts, test_ids = prepare_timeseries_from_dataset(self.test_dataset)

        self.train_ds, self.train_ids = train_ts, train_ids
        self.test_ds, self.test_ids = test_ts, test_ids

        def prepare_concat_ts_array(
            timeseries: list,
        ) -> np.ndarray:
            def copy_timeseries_to_zero_array(
                series_length=self.seq_len, unsqueeze=False
            ):
                shape = [len(timeseries), series_length]
                zero_array = np.zeros(shape)
                for i, single_array in enumerate(timeseries):
                    zero_array[i, : timeseries[i].shape[0]] = timeseries[i][
                        :series_length
                    ]
                if unsqueeze:
                    return np.expand_dims(zero_array, axis=2)
                else:
                    return zero_array

            timeseries_array = copy_timeseries_to_zero_array()
            concat_time_series = np.expand_dims(
                np.concatenate(timeseries_array), axis=1
            )
            return concat_time_series

        self.train_concat_ts = prepare_concat_ts_array(train_ts)
        self.train_concat_ts_df = pd.DataFrame(
            self.train_concat_ts, columns=["value"]
        )  # Caching, so we don't need to recompute it every time

        self.test_concat_ts = prepare_concat_ts_array(test_ts)

        self.scaler = StandardScaler()
        scaled_train_concat_ts = self.scaler.fit_transform(self.train_concat_ts)
        scaled_test_concat_ts = self.scaler.transform(self.test_concat_ts)

        scaled_train_ts = scaled_train_concat_ts.reshape(len(train_ts), -1)
        scaled_test_ts = scaled_test_concat_ts.reshape(len(test_ts), -1)

        # Caching for future debugging
        self.scaled_train_ts = scaled_train_ts
        self.scaled_test_ts = scaled_test_ts

        if self.flag == "train":
            self.timeseries = train_ts
            self.scaled_ts = scaled_train_ts
        else:
            self.timeseries = test_ts
            self.scaled_ts = scaled_test_ts

        self.a = 1

    def get_train_data(self):
        # Needed for "Multi" dataloader :)
        return self.train_concat_ts_df

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        # We fetch the 'original' time-series to then use it to reshape the 'scaled' time-series accordingly
        sampled_orig_timeseries = self.timeseries[index]

        # We use the scaled time-series
        sampled_timeseries = self.scaled_ts[index][: len(sampled_orig_timeseries)]

        cut_point = np.random.randint(
            low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
            high=len(sampled_timeseries),
            size=1,
        )[0]

        insample_window = sampled_timeseries[
            max(0, cut_point - self.seq_len) : cut_point
        ]
        insample[-len(insample_window) :, 0] = insample_window
        insample_mask[-len(insample_window) :, 0] = 1.0
        outsample_window = sampled_timeseries[
            cut_point
            - self.label_len : min(len(sampled_timeseries), cut_point + self.pred_len)
        ]
        outsample[: len(outsample_window), 0] = outsample_window
        outsample_mask[: len(outsample_window), 0] = 1.0

        # Adapting masks to the correct size
        mask_length = 4
        insample_mask = insample_mask.repeat(mask_length, axis=1)
        outsample_mask = outsample_mask.repeat(mask_length, axis=1)

        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len :]
            insample[i, -len(ts) :] = ts_last_window
            insample_mask[i, -len(ts) :] = 1.0
        return insample, insample_mask


if __name__ == "__main__":
    root_path = os.path.join(FileUtils.project_root_dir(), "dataset")
    site_id = "S44"
    data_path = "NYC_data_15min.csv"
    flag = "train"
    seq_len = 24
    label_len = 20
    pred_len = 4
    features = "S"  # NOTE: only supports 'S' for now
    target = (None,)
    timeenc = 1
    freq = "h"  # TODO: make 15 min

    args = Namespace(
        is_training=1,
        model_id="v2_ETTh1_PatchTST_seq_len_96_hf",
        model="PatchTST",
        should_log=True,
        name="20240111_heatflex",
        data="Multi",
        site_id="nist2013,S22",
        target_site_id="nist2013",
        root_path="/home/jonas/repos/neoebm/dataset",
        data_path="NYC_data_15min.csv",
        features="S",
        feature=None,
        target="OT",
        freq="h",
        checkpoints="./checkpoints/",
        seq_len=100,
        label_len=96,
        pred_len=4,
        individual=False,
        embed_type=0,
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=64,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=32,
        moving_avg=25,
        factor=3,
        distil=True,
        dropout=0.05,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        do_predict=False,
        num_workers=0,
        itr=1,
        train_epochs=5,
        batch_size=32,
        patience=3,
        learning_rate=0.001,
        des="experiment",
        loss="mse",
        lradj="type1",
        use_amp=False,
        ebm_samples=256,
        ebm_epochs=5,
        ebm_model_name="patch_tst_mlp_concat",
        ebm_hidden_size=16,
        ebm_num_layers=1,
        ebm_decoder_num_layers=4,
        ebm_predictor_size=96,
        ebm_decoder_size=96,
        ebm_optim_lr=0.001,
        ebm_inference_optim_lr=0.001,
        ebm_inference_optim_steps=50,
        ebm_inference_batch_size=32,
        ebm_validate_during_training_step=10,
        ebm_training_method="nce",
        ebm_seed=2023,
        ebm_training_strategy="train_y_and_xy_together",
        ebm_cd_step_size=0.1,
        ebm_cd_num_steps=10,
        ebm_cd_alpha=0.9,
        ebm_cd_sched_rate=1.0,
        use_gpu=False,
        gpu=0,
        use_multi_gpu=False,
        devices="0,1,2,3",
        test_flop=False,
        output_parent_path="/home/jonas/repos/neoebm/output_2024/20240111",
        experiment_only_on_given_model_path="None",
        only_rerun_inference=0,
        only_output_model_params=0,
        force_retrain_orig_model=False,
        force_retrain_y_enc=False,
        force_retrain_xy_dec=False,
        ebm_margin_loss=-1.0,
        version="Fourier",
        mode_select="random",
        modes=64,
        top_k=5,
        num_kernels=6,
        multi_data="ETTh1,ETTh2",
        multi_data_path="ETTh1.csv,ETTh2.csv",
    )

    seasonal_patterns = "Monthly"
    dataset_m4 = Dataset_M4(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=seasonal_patterns,
    )

    batch_size = 16
    drop_last = True
    num_workers = 0
    shuffle_flag = True

    train_loader = DataLoader(
        dataset_m4,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    print(f"Length of M4 train loader = {len(train_loader)}")

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        batch_x, batch_y, batch_x_mark, batch_y_mark = (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
        )
        break
