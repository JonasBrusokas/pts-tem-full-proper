import os
from pathlib import Path

from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Pred,
)
from data_provider.heatflex_data_loader import Dataset_HeatflexMulti
from data_provider.m4_data_loader import Dataset_M4
from data_provider.multi_data_loader import Dataset_Multi
from data_provider.ran_data_loader import Dataset_RAN  # Add this import

ran_v2_label_base = "ran_v2"
ran_v2_label_single_user = "ran_v2_s"

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "HeatFlex": Dataset_HeatflexMulti,  # NOTE: we will use "multi" for everything for the caching and other
    # "HeatFlexSingle": Dataset_Heatflex,
    "HeatFlexMulti": Dataset_HeatflexMulti,
    "Multi": Dataset_Multi,
    "RanMulti": Dataset_Multi,  # <<< Handled separately!
    # Ported M4 dataset
    "m4": Dataset_M4,
    "ran_v1": Dataset_RAN,  # NOTE: deprecated
    "ran_v1_a": Dataset_RAN,  # NOTE: deprecated
    ran_v2_label_base: Dataset_RAN,
}


def extract_data_and_data_path_tuples(
    multi_data_string: str, multi_data_path_string: str
):
    data_list = list(
        map(lambda data_str: data_str.strip(), multi_data_string.split(","))
    )
    data_path_list = list(
        map(
            lambda data_path_str: data_path_str.strip(),
            multi_data_path_string.split(","),
        )
    )
    if len(data_path_list) != len(data_list):
        raise ValueError(f"The length of data_list and multi_data_path does not match!")
    return list(zip(data_list, data_path_list))


def data_provider(
    args,
    flag,
    override_batch_size=None,
    override_data_path=None,
    override_scaler=None,
    override_target_site_id=None,
    override_dataset_tuples=None,
):
    # We use this for overriding larger datasets (such as m4 and RAN)

    def fetch_data_to_use():
        data_to_use = None
        if args.data.startswith("m4"):
            data_to_use = data_dict["m4"]
        elif args.data.startswith("ran"):
            data_to_use = data_dict[ran_v2_label_base]
        else:
            data_to_use = data_dict[args.data]
        return data_to_use

    Data = fetch_data_to_use()
    timeenc = 0 if args.embed != "timeF" else 1

    if override_data_path is not None:
        if not Path(os.path.join(args.root_path, override_data_path)).exists():
            raise ValueError(f"Data at path {override_data_path} does not exist!")
        data_path = override_data_path
    else:
        data_path = args.data_path

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data in ["ETTh1", "ETTh2"]:
        data_set = Data(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            override_scaler=override_scaler,
        )
    elif args.data in ["custom"]:
        data_set = Data(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )
    elif args.data in ["HeatFlexMulti"]:
        site_id_list = list(
            map(lambda site_id_str: site_id_str.strip(), args.site_id.split(","))
        )
        if override_target_site_id is not None:
            site_id_list = [override_target_site_id]
            map(
                lambda site_id_str: site_id_str.strip(),
                override_target_site_id.split(","),
            )
            print(f"*** HeatFlexMulti overriden parsed site ids to: {site_id_list}")
        else:
            print(f"HeatFlexMulti Successfully parsed site ids: {site_id_list}")

        data_set = Dataset_HeatflexMulti(
            root_path=args.root_path,
            site_id_list=site_id_list,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            override_scaler=override_scaler,
        )
    elif args.data in ["HeatFlex"]:
        # data_set = Dataset_Heatflex(
        #     root_path=args.root_path,
        #     site_id=args.site_id,
        #     data_path=data_path,
        #     flag=flag,
        #     size=[args.seq_len, args.label_len, args.pred_len],
        #     features=args.features,
        #     timeenc=timeenc,
        #     freq=freq,
        #     override_scaler=override_scaler,
        # )
        data_set = Dataset_HeatflexMulti(
            root_path=args.root_path,
            site_id_list=[args.site_id],
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            override_scaler=override_scaler,
        )
    elif args.data in ["Multi", "RanMulti"]:
        data_and_path_tuples = extract_data_and_data_path_tuples(
            multi_data_string=args.multi_data,
            multi_data_path_string=args.multi_data_path,
        )
        if override_dataset_tuples is not None:
            print(
                f"DatasetMulti: using overriden dataset tuples: {override_dataset_tuples}"
            )
            if isinstance(override_dataset_tuples, tuple):
                data_and_path_tuples = [override_dataset_tuples]
            else:
                data_and_path_tuples = override_dataset_tuples
        else:
            print(f"Using found data and path tuples: {data_and_path_tuples}")

        should_validate_univariate_inputs = True
        if args.data == "RanMulti":
            should_validate_univariate_inputs = False
            illegal_data_configs = list(
                filter(
                    lambda data_and_path_tuple: not data_and_path_tuple[0].startswith(
                        "ran"
                    ),
                    data_and_path_tuples,
                )
            )
            if len(illegal_data_configs) != 0:
                raise ValueError(
                    f"Illegal RanMulti data configs (must begin with 'ran'): {illegal_data_configs}"
                )

        data_set = Dataset_Multi(
            root_path=args.root_path,
            data_and_path_tuples=data_and_path_tuples,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            override_scaler=override_scaler,
            args=args,
            data_provider_lambda=data_provider,
            validate_univariate_features=should_validate_univariate_inputs,
        )
    elif args.data.startswith("m4"):
        data_split = args.data.split("_")
        if len(data_split) != 2:
            raise ValueError(f"args.data = '{args.data}' is invalid!")
        seasonal_patterns = data_split[1]
        data_set = Dataset_M4(
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
    elif args.data.startswith("ran_"):  # Add this condition
        user_ids_to_use = None
        labels_to_use = None
        if args.data == "ran_v1":
            user_ids_to_use = Dataset_RAN.SMALL_USER_IDS_TO_USE
            labels_to_use = Dataset_RAN.SMALL_LABELS_TO_USE
            print(
                f"Using smaller subset of existing user ids and labels. user_ids={user_ids_to_use} labels={labels_to_use}"
            )
            raise ValueError(f"Unsupported data: {args.data}")
        elif args.data in ["ran_v1_a"]:
            user_ids_to_use = Dataset_RAN.SMALL_USER_IDS_TO_USE
            labels_to_use = Dataset_RAN.DEFAULT_LABELS_TO_USE
            print(
                f"Using smaller subset of existing user ids, but with ALL labels. user_ids={user_ids_to_use} labels={labels_to_use}"
            )
            raise ValueError(f"Unsupported data: {args.data}")
        elif args.data.startswith(ran_v2_label_single_user):
            data_postfix_string = args.data[len(ran_v2_label_single_user) :]

            def relabel(label: str) -> str:
                alt_label = label
                if label == "web":
                    alt_label = "Web Browsing"
                return alt_label

            # Relabel and then filter empty labels
            labels_to_use = list(
                filter(
                    lambda label_str: label_str is not None and label_str != "",
                    map(
                        lambda label_str: relabel(label_str),
                        data_postfix_string.split("_"),
                    ),
                )
            )
            # If empty, use 'None' as the default
            labels_to_use = None if len(labels_to_use) == 0 else labels_to_use
            user_ids_to_use = Dataset_RAN.SMALL_USER_IDS_TO_USE
            print(
                f"Using smaller subset of existing user ids and labels. user_ids={user_ids_to_use} labels={labels_to_use}"
            )

        data_set = Dataset_RAN(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            timeenc=timeenc,
            freq=freq,
            user_ids_to_use=user_ids_to_use,
            labels_to_use=labels_to_use,
            override_scaler=override_scaler,  # Pass the override_scaler here
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.data}")
        data_set = Data(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size if override_batch_size is None else override_batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader


def get_data_from_provider(
    args,
    flag,
    override_batch_size=None,
    override_data_path=None,
    override_scaler=None,
    override_target_site_id=None,
    override_dataset_tuples=None,
):
    data_set, data_loader = data_provider(
        args=args,
        flag=flag,
        override_batch_size=override_batch_size,
        override_data_path=override_data_path,
        override_scaler=override_scaler,
        override_target_site_id=override_target_site_id,
        override_dataset_tuples=override_dataset_tuples,
    )
    return data_set, data_loader


# %%
#
data = "m4_Yearly"
data_split = data.split("_")
if len(data_split) != 2:
    raise ValueError(f"args.data = '{data}' is invalid!")
seasonal_patterns = data_split[1]
