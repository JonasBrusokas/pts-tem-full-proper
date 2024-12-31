from argparse import Namespace

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader

from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour
from utilz import *


# NOTE: this is a generalized case of the HeatFlexMulti dataset
class Dataset_Multi(Dataset_Custom):
    scaler_cache = {}
    dataset_cache = {}

    @staticmethod
    def add_scaler_to_cache(key_joined, scaler_to_cache):
        Dataset_Multi.scaler_cache[key_joined] = scaler_to_cache

    @staticmethod
    def fetch_scaler_from_cache(key_joined):
        return Dataset_Multi.scaler_cache[key_joined]

    @staticmethod
    def fetch_dataset_from_cache(key_joined):
        (
            concat_dataset_from_cache,
            site_datasets_from_cache,
        ) = Dataset_Multi.dataset_cache[key_joined]
        return concat_dataset_from_cache, site_datasets_from_cache

    @staticmethod
    def add_dataset_to_cache(
        key_joined, concat_dataset_to_cache, site_datasets_to_cache
    ):
        print(f"Adding {key_joined}: {len(concat_dataset_to_cache)}")
        Dataset_Multi.dataset_cache[key_joined] = (
            concat_dataset_to_cache,
            site_datasets_to_cache,
        )

    def __init__(
        self,
        root_path: str,
        data_and_path_tuples: [(str, str)],
        flag: str = "train",
        size=None,
        features="S",  # TODO: For now we support only "S"
        data_path="ETTh1.csv",
        target=None,  # NOTE: target doesn't really make sense anymore, since every dataset might have a different target
        scale=True,
        timeenc=0,
        freq="h",
        override_scaler=None,
        args=None,
        data_provider_lambda=None,
        validate_univariate_features=True,
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq
        )
        self.override_scaler = override_scaler

        if features != "S" and validate_univariate_features:
            raise ValueError("NB: Only support features 'S' for MULTI LOADER")
        if args is None:
            raise ValueError("Args not provided! Necessary for MULTI LOADER")

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
        self.target = None
        print(
            f"MULTI LOADER: Y target overriden (given: ${target}) to '${self.target}'"
        )

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.data_and_path_tuples = data_and_path_tuples

        self.root_path = root_path
        self.data_path = data_path

        def setup_one_dataset(
            data_and_path_tuple: [(str, str)],
            orig_args,
            flag=flag,
            fitted_scaler=None,
        ):
            args_for_one_dataset = Namespace(**vars(orig_args))
            args_for_one_dataset.data = data_and_path_tuple[0]
            args_for_one_dataset.data_path = data_and_path_tuple[1]
            dataset, dataloader = data_provider_lambda(
                args=args_for_one_dataset,
                flag=flag,
                override_scaler=fitted_scaler,
            )
            return dataset

        data_path_list = list(
            map(
                lambda data_path_tuple: f"{data_path_tuple[0]}_:_{data_path_tuple[1]}",
                data_and_path_tuples,
            )
        )
        data_path_joined_key = ",".join(data_path_list)
        if override_scaler is not None:
            data_path_joined_key += "___overriden_scaler"

        dataset_joined_key_and_flag = f"{data_path_joined_key}_{flag}"

        if dataset_joined_key_and_flag not in Dataset_Multi.dataset_cache:
            print(
                f"Datasets for MULTI LOADER ({dataset_joined_key_and_flag}) dataset not found!"
            )
            if data_path_joined_key not in Dataset_Multi.scaler_cache:
                print(
                    f"Scaler for MULTI LOADER ({data_path_joined_key}) dataset not found"
                )
                if override_scaler is None:
                    # First, create datasets without a scaler to collect training data
                    self.orig_datasets = list(
                        map(
                            lambda data_and_path_tuple: setup_one_dataset(
                                data_and_path_tuple=data_and_path_tuple,
                                orig_args=args,
                                fitted_scaler=None,
                            ),
                            data_and_path_tuples,
                        )
                    )
                    # Collect training data
                    self.orig_concat_dataset_train = pd.concat(
                        list(
                            map(
                                lambda dataset: dataset.get_train_data(),
                                self.orig_datasets,
                            )
                        )
                    )
                    # Fit scaler
                    self.concat_scaler = StandardScaler().fit(
                        self.orig_concat_dataset_train
                    )
                else:
                    self.concat_scaler = override_scaler
                Dataset_Multi.add_scaler_to_cache(
                    data_path_joined_key, scaler_to_cache=self.concat_scaler
                )
                print(
                    f"Added {'override' if override_scaler else 'new'} scaler to cache"
                )
            else:
                print(
                    f"Scaler for MULTI LOADER ({data_path_joined_key}) dataset FOUND!"
                )
                self.concat_scaler = Dataset_Multi.fetch_scaler_from_cache(
                    data_path_joined_key
                )

            # Now create the datasets with the appropriate scaler
            self.datasets = list(
                map(
                    lambda data_and_path_tuple: setup_one_dataset(
                        data_and_path_tuple=data_and_path_tuple,
                        orig_args=args,
                        fitted_scaler=self.concat_scaler,
                    ),
                    data_and_path_tuples,
                )
            )
            self.concat_dataset = ConcatDataset(self.datasets)
            self.add_dataset_to_cache(
                key_joined=dataset_joined_key_and_flag,
                concat_dataset_to_cache=self.concat_dataset,
                site_datasets_to_cache=self.datasets,
            )
        else:
            print(
                f"Datasets for MULTI LOADER ({dataset_joined_key_and_flag}) dataset FOUND!"
            )
            (
                self.concat_dataset,
                self.datasets,
            ) = self.fetch_dataset_from_cache(dataset_joined_key_and_flag)
            self.concat_scaler = Dataset_Multi.fetch_scaler_from_cache(
                data_path_joined_key
            )

        # For compatibility!
        self.scaler = self.concat_scaler

    def __read_data__(self):
        # Just a NOP method
        pass

    def __getitem__(self, index):
        return self.concat_dataset.__getitem__(index)

    def __len__(self):
        return self.concat_dataset.__len__()

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
    features = "S"  # NOTE: only supports 'S' for now
    target = (None,)
    timeenc = 1
    freq = "h"  # TODO: make 15 min

    mock_args_old = Namespace(
        **(
            {
                "root_path": root_path,
                "data_path": data_path,
                "flag": flag,
                "seq_len": seq_len,
                "label_len": label_len,
                "pred_len": pred_len,
                "features": features,
                "target": target,
                "timeenc": timeenc,
                "freq": freq,
            }
        )
    )

    mock_args = Namespace(
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

    example_dataset = Dataset_ETT_hour(
        root_path=root_path,
        data_path="ETTh1.csv",
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target="OT",
        timeenc=timeenc,
        freq=freq,
    )
    example_dataset_2 = Dataset_ETT_hour(
        root_path=root_path,
        data_path="ETTh2.csv",
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target="OT",
        timeenc=timeenc,
        freq=freq,
    )

    from data_factory import data_provider

    data_and_path_tuples = [("ETTh1", "ETTh1.csv"), ("ETTh2", "ETTh2.csv")]
    dataset = Dataset_Multi(
        root_path=root_path,
        data_and_path_tuples=data_and_path_tuples,
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=timeenc,
        freq=freq,
        args=mock_args,
        data_provider_lambda=data_provider,
    )

    def iterate_through_dataset(dataset):
        batch_size = 1
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

        length = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
            train_loader
        ):
            batch_x, batch_y, batch_x_mark, batch_y_mark = (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            )
            length += 1

        print(f"Length: {length}")
        return length

    iterate_through_dataset(example_dataset)
    iterate_through_dataset(example_dataset_2)
    iterate_through_dataset(dataset)

    logs_of_dataset_length = f"""
Multi dataset {len(dataset)}
example dataset {len(example_dataset)}
example dataset 2 {len(example_dataset_2)}
    """
    print(logs_of_dataset_length)

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

    dataset_multi = Dataset_Multi(
        root_path=root_path,
        data_and_path_tuples=data_and_path_tuples,
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=timeenc,
        freq=freq,
        args=mock_args,
        data_provider_lambda=data_provider,
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

    site_dataset = dataset_multi.datasets[0]
