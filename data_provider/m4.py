# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
M4 Dataset
"""
import itertools
import logging
import pathlib
import sys
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from urllib import request

import numpy as np
import pandas as pd

from utilz import *


####


def _download(url: str, file_path: str) -> None:
    """
    Download a file to the given path.

    :param url: URL to download
    :param file_path: Where to download the content.
    """

    def progress(count, block_size, total_size):
        progress_pct = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write(
            "\rDownloading {} to {} {:.1f}%".format(url, file_path, progress_pct)
        )
        sys.stdout.flush()

    if not os.path.isfile(file_path):
        opener = request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        request.install_opener(opener)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write("\n")
        sys.stdout.flush()
        file_info = os.stat(f)
        logging.info(
            f"Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes."
        )
    else:
        file_info = os.stat(file_path)
        logging.info(f"File already exists: {file_path} {file_info.st_size} bytes.")


def _url_file_name(url: str) -> str:
    """
    Extract file name from url.

    :param url: URL to extract file name from.
    :return: File name.
    """
    return url.split("/")[-1] if len(url) > 0 else ""


####

FREQUENCIES = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
URL_TEMPLATE = (
    "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/{}/{}-{}.csv"
)

TRAINING_DATASET_URLS = [
    URL_TEMPLATE.format("Train", freq, "train") for freq in FREQUENCIES
]
TEST_DATASET_URLS = [URL_TEMPLATE.format("Test", freq, "test") for freq in FREQUENCIES]
INFO_URL = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/M4-info.csv"
NAIVE2_FORECAST_URL = "https://github.com/M4Competition/M4-methods/raw/master/Point%20Forecasts/submission-Naive2.rar"

COMMON_DATASETS_PATH = os.path.join(FileUtils.project_root_dir(), "dataset")
DATASET_PATH = os.path.join(COMMON_DATASETS_PATH, "m4")
print(f"M4 dataset path: {DATASET_PATH}")

TRAINING_DATASET_FILE_PATHS = [
    os.path.join(DATASET_PATH, _url_file_name(url)) for url in TRAINING_DATASET_URLS
]
TEST_DATASET_FILE_PATHS = [
    os.path.join(DATASET_PATH, _url_file_name(url)) for url in TEST_DATASET_URLS
]
INFO_FILE_PATH = os.path.join(DATASET_PATH, _url_file_name(INFO_URL))
NAIVE2_FORECAST_FILE_PATH = os.path.join(DATASET_PATH, "submission-Naive2.csv")


TRAINING_DATASET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, "training.npz")
TEST_DATASET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, "test.npz")


@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True) -> "M4Dataset":
        try:
            M4Dataset.download()  # Try to download files if needed!
        except Exception as e:
            print("Exception during downloading M4Dataset.")
            if not M4Dataset.check_if_m4_files_are_downloaded():
                raise ValueError(
                    f"Missing M4 files, cannot download automatically. Please download them manually"
                )
        """
        Load cached dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        m4_info = pd.read_csv(INFO_FILE_PATH)
        return M4Dataset(
            ids=m4_info.M4id.values,
            groups=m4_info.SP.values,
            frequencies=m4_info.Frequency.values,
            horizons=m4_info.Horizon.values,
            values=np.load(
                TRAINING_DATASET_CACHE_FILE_PATH
                if training
                else TEST_DATASET_CACHE_FILE_PATH,
                allow_pickle=True,
            ),
        )

    @staticmethod
    def check_if_m4_files_are_downloaded() -> bool:
        def file_string(freq, flag):
            return f"{freq}-{flag}.csv"

        flags = ["train", "test"]

        # Generate necessary file names
        necessary_freq_flag_combinations = itertools.product(FREQUENCIES, flags)
        necessary_files = list(
            map(
                # Generate file string from freq and flag
                lambda freq_flag_tuple: file_string(
                    freq=freq_flag_tuple[0], flag=freq_flag_tuple[1]
                ),
                necessary_freq_flag_combinations,
            )
        )

        # Check for missing files
        files_in_m4_data_directory = os.listdir(DATASET_PATH)
        missing_files = list(
            filter(
                lambda necessary_file: not (
                    necessary_file in files_in_m4_data_directory
                ),
                necessary_files,
            )
        )
        if len(missing_files) != 0:
            print(f"Missing following files: {missing_files}")
            return False

        return True

    @staticmethod
    def download() -> None:
        """
        Download M4 dataset if doesn't exist.
        """
        if M4Dataset.check_if_m4_files_are_downloaded():
            # print(f"All M4 files already found! No need to download")
            return

        _download(INFO_URL, INFO_FILE_PATH)
        m4_ids = pd.read_csv(INFO_FILE_PATH).M4id.values

        def build_cache(files: str, cache_path: str) -> None:
            try:
                timeseries_dict = OrderedDict(list(zip(m4_ids, [[]] * len(m4_ids))))
                logging.info(f"Caching {files}")
                for train_csv in glob(os.path.join(DATASET_PATH, files)):
                    dataset = pd.read_csv(train_csv)
                    dataset.set_index(dataset.columns[0], inplace=True)
                    for m4id, row in dataset.iterrows():
                        values = row.values
                        timeseries_dict[m4id] = values[~np.isnan(values)]
                # np.array(list(timeseries_dict.values())).dump(cache_path)
                # NOTE: the above line "does not fly" anymore - https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
                np.asarray(list(timeseries_dict.values()), dtype="object").dump(
                    cache_path
                )
            except Exception as e:
                raise e

        for url, path in zip(TRAINING_DATASET_URLS, TRAINING_DATASET_FILE_PATHS):
            _download(url, path)
        build_cache("*-train.csv", TRAINING_DATASET_CACHE_FILE_PATH)

        for url, path in zip(TEST_DATASET_URLS, TEST_DATASET_FILE_PATHS):
            _download(url, path)
        build_cache("*-test.csv", TEST_DATASET_CACHE_FILE_PATH)


@dataclass()
class M4Meta:
    seasonal_patterns = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        "Yearly": 6,
        "Quarterly": 8,
        "Monthly": 18,
        "Weekly": 13,
        "Daily": 14,
        "Hourly": 48,
    }
    frequency_map = {
        "Yearly": 1,
        "Quarterly": 4,
        "Monthly": 12,
        "Weekly": 1,
        "Daily": 1,
        "Hourly": 24,
    }
    history_size = {
        "Yearly": 1.5,
        "Quarterly": 1.5,
        "Monthly": 1.5,
        "Weekly": 10,
        "Daily": 10,
        "Hourly": 10,
    }  # from interpretable.gin


def load_m4_info() -> pd.DataFrame:
    """
    Load M4Info file.

    :return: Pandas DataFrame of M4Info.
    """
    return pd.read_csv(INFO_FILE_PATH)


if __name__ == "__main__":
    M4Dataset.download()
    print(f"Done!")
