# Based on https://gitlab.com/JohnLogic/ct-reconstruction/-/blob/master/utils.py
# from my MSc thesis project
import base64
# Based on https://gitlab.com/JohnLogic/heat-pump-first-paper-project/-/blob/master/utils.py
# from first PhD paper "HeatFlex: Machine learning based data-driven flexibility prediction for individual heat pumps"
import functools
import hashlib
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import git


class FileUtils:
    DEFAULT_TEMP_FOLDER = "./tmp"

    @classmethod
    def apth(cls, path_string) -> str:
        path_obj = Path(path_string)
        absolute_path = str(path_obj.resolve())
        return cls.pth(absolute_path)

    @classmethod
    def pth(cls, path_string) -> str:
        return os.path.join(*path_string.split("/"))

    @classmethod
    def create_dir(cls, path_string: str, fail_if_exists: bool = False) -> str:
        path_obj = Path(path_string)
        path_obj.mkdir(parents=True, exist_ok=not fail_if_exists)
        return str(path_obj.resolve())

    @classmethod
    def file_name(cls, path_string: str, with_extension: bool = False) -> str:
        path_obj = Path(path_string)
        return path_obj.stem if not with_extension else path_obj.name

    @classmethod
    def home_dir(cls) -> str:
        return str(Path.home())

    @classmethod
    def project_root_dir(cls) -> str:
        # NOTE: you need to change the definition according to the directory of 'utils.py'
        return str(Path(__file__).parent)

    @classmethod
    def calculate_hash(cls, file_path: str, buffer_size: int = 65536) -> str:
        file_hash = (
            hashlib.sha256()
        )  # Create the hash object, can use something other than `.sha256()` if you wish
        with open(file_path, "rb") as f:  # Open the file to read it's bytes
            fb = f.read(
                buffer_size
            )  # Read from the file. Take in the amount declared above
            while len(fb) > 0:  # While there is still data being read from the file
                file_hash.update(fb)  # Update the hash
                fb = f.read(buffer_size)  # Read the next block from the file

        return file_hash.hexdigest()
        # print(file_hash.hexdigest())  # Get the hexadecimal digest of the hash

    @classmethod
    def write_string_to_file(cls, file_path: str, string_to_write: str):
        with open(file_path, "w") as file:
            file.write(string_to_write)

    @staticmethod
    def sanitize_filename(s: str) -> str:
        """
        Sanitize the given string to create a valid filename for Linux and Mac filesystems.

        This function:
        1. Normalizes unicode characters
        2. Removes or replaces invalid characters
        3. Trims leading/trailing whitespace and periods
        4. Ensures the filename is not empty or just dots
        5. Truncates the filename if it's too long

        Args:
            s (str): The input string to sanitize

        Returns:
            str: A sanitized string safe to use as a filename
        """
        # Normalize unicode characters
        s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")

        # Remove or replace invalid characters
        s = re.sub(r"[^\w\s-]", "_", s)

        # Replace spaces with underscores
        s = re.sub(r"\s+", "_", s)

        # Remove leading/trailing whitespace and periods
        s = s.strip().strip(".")

        # Ensure the filename is not empty or just dots
        if not s or s == ".":
            s = "_"

        # Truncate if the filename is too long (255 bytes is a common limit)
        max_length = 255
        if len(s.encode("utf-8")) > max_length:
            s = s.encode("utf-8")[:max_length].decode("utf-8", "ignore").rstrip()
            raise ValueError(
                f"Filename too long ({len(s.encode('utf-8'))} > {max_length}): '{s}'"
            )

        return s


class DateUtils:
    DATE_FORMAT = "%Y-%m-%d"
    TIME_FORMAT = "%H-%M-%S"
    DATETIME_FORMAT = "{}_{}".format(DATE_FORMAT, TIME_FORMAT)

    @classmethod
    def now(cls):
        now = datetime.now()
        return now

    @classmethod
    def formatted_datetime_now(cls):
        return datetime.now().strftime(cls.DATETIME_FORMAT)


class ListUtils:
    @classmethod
    def natural_sort(cls, list_to_sort: [str]) -> [str]:
        """Sort the given list of strings in the way that humans expect."""
        import copy
        import re

        copied_list = copy.deepcopy(list_to_sort)
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        copied_list.sort(key=alphanum_key)
        return copied_list

    @classmethod
    # Provides with a subdict, containing the key intersection having different values
    # Useful when inspecting what values were change between same structured diffs
    # Returns NEW (dict2) values
    def delta_dict(cls, dict1: dict, dict2: dict) -> (dict, dict):
        key_intersection = set(dict1.keys()).intersection(set(dict2.keys()))
        delta_dict_old, delta_dict_new = {}, {}
        for key in key_intersection:
            if dict1[key] != dict2[key]:
                delta_dict_old[key] = dict1[key]
                delta_dict_new[key] = dict2[key]
        return delta_dict_old, delta_dict_new

    @classmethod
    def flatten_list(cls, list_to_flatten: list) -> list:
        return [item for sublist in list_to_flatten for item in sublist]

    def permute_dict(dictionary: dict) -> [dict]:
        # Inner method to avoid exposing the purely output parameter 'output_dictionary_list'
        def _permute_dict(
            dictionary: dict, output_dictionary_list: [dict] = [{}]
        ) -> [dict]:
            if len(dictionary) == 0:
                return output_dictionary_list

            dictionary_copy = dictionary.copy()
            sorted_keys = ListUtils.natural_sort(list(dictionary_copy.keys()))
            first_key = sorted_keys[0]
            # rest_keys = sorted_keys[1:]
            first_key_values = dictionary_copy.pop(first_key)

            def add_to_dict(dictionary: dict, value: dict) -> dict:
                idict = dictionary.copy()
                idict.update(value)
                return idict

            for idx, _ in enumerate(output_dictionary_list):
                output_dictionary_list[idx] = list(
                    map(
                        lambda value_for_key: add_to_dict(
                            output_dictionary_list[idx], {first_key: value_for_key}
                        ),
                        first_key_values,
                    )
                )

            return _permute_dict(
                dictionary=dictionary_copy,
                output_dictionary_list=ListUtils.flatten_list(output_dictionary_list),
            )

        return _permute_dict(dictionary)

    @classmethod
    def list_to_compact_string(cls, lst: [Any]):
        string = ""
        for item in lst:
            string += f",{item}"
        return string

    @classmethod
    def split_list_sequentially(cls, lst: [Any], number_of_buckets: int):
        # NOTE: DANGER!!!! lists get copied as references!!!
        # buckets = [([])] * number_of_buckets
        buckets = []
        for _ in range(number_of_buckets):
            buckets.append([])

        for idx, item in enumerate(lst):
            bucket_idx = idx % number_of_buckets
            buckets[bucket_idx].append(item)
        return buckets

    @classmethod
    def select_ith_from_2d_list(cls, list_2d, i):
        return list(map(lambda x: x[i], list_2d))

    @staticmethod
    def get_dict_key_from_item(dct: dict, item: Any) -> [Any]:
        list_of_idx = list(filter(lambda idx: idx == item, list(dct.values())))
        if len(list_of_idx) > 1:
            raise ValueError(
                f"More than one item in the dictionary with value '{item}'"
            )

        keys = list(dct.keys())[list_of_idx[0]]
        return keys


class FunkyUtils:
    @classmethod
    def len_iterable(cls, iterable):
        # Based on: https://stackoverflow.com/questions/31011631/python-2-3-object-of-type-zip-has-no-len
        return sum(1 for _ in iterable)

    @classmethod
    def grab_first(cls, iterable):
        for item in iterable:
            return item


class ArgParams:
    def __init__(self, args):
        self.args = args
        self.params = {}

    def str_arg(self, name, **kwargs):
        return self.dtype_arg(name, str, **kwargs)

    def int_arg(self, name, **kwargs):
        return self.dtype_arg(name, int, **kwargs)

    def float_arg(self, name, **kwargs):
        return self.dtype_arg(name, float, **kwargs)

    def bool_arg(self, name, required=False, default: Optional[bool] = None):
        # return self.dtype_arg(name, bool, **kwargs) # TODO: fix proper, remove code rep
        if (name not in self.args) or self.args[name] is None:
            if required and default is None:
                raise ValueError(f"Key '{name}' is required and not present in args!")
            else:
                # Sets to default value
                self.params[name] = default
        else:
            self.params[name] = True if (self.args[name] in ["true", "True"]) else False
        return self.params[name]

    def int_list(self, name: str, required: bool = False, default=None):
        ...

    def dtype_arg(self, name, dtype, required: bool = False, default=None):
        if (name not in self.args) or self.args[name] is None:
            if required and default is None:
                raise ValueError(f"Key '{name}' is required and not present in args!")
            else:
                # Sets to default value
                self.params[name] = default
        else:
            self.params[name] = dtype(self.args[name])
        return self.params[name]

    def custom_arg(self, name, lambda_arg_to_val, required: bool = False):
        self.params[name] = lambda_arg_to_val(self.args[name])
        return self.params[name]

    def get_params(self):
        return self.params


class ArgParseUtils:
    @staticmethod
    def copy_args(args):
        from argparse import Namespace

        return Namespace(**vars(args))


class RepoUtils:
    @classmethod
    def getRepoHash(cls):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha


class TypeUtils:
    @classmethod
    def check_args_for_existing(cls, arguments: [Any]) -> bool:
        if None in arguments:
            raise ValueError(f"Missing mandatory arguments: {arguments}")
        return True

    @classmethod
    def check_dict_keys_for_existing(cls, somed: dict, keys: [str]) -> bool:
        key_exist_list = list(map(lambda key: key in somed, keys))
        all_in = functools.reduce(
            lambda curr_val, elem: curr_val and elem, key_exist_list
        )
        if not all_in:
            error_string = "Keys: "
            for idx, key in enumerate(keys):
                if not key_exist_list[idx]:
                    error_string += f"'{keys[idx]}' "
            error_string += "are missing from given dictionary."
            raise ValueError(f"{error_string}")
        return True


class TEMUtils:
    @staticmethod
    def get_fixed_length_hash(input_string):
        encoded_string = input_string.encode("utf-8")
        sha256_hash = hashlib.sha256()
        sha256_hash.update(encoded_string)
        hex_digest = sha256_hash.hexdigest()
        return hex_digest

    @staticmethod
    def get_base64_hash(input_string):
        encoded_string = input_string.encode("utf-8")
        sha256_hash = hashlib.sha256()
        sha256_hash.update(encoded_string)
        binary_digest = sha256_hash.digest()
        base64_hash = base64.urlsafe_b64encode(binary_digest).decode("utf-8")
        base64_hash = base64_hash.rstrip("=")
        return base64_hash

    @staticmethod
    def generate_multi_dataset_name(multi_data_path, multi_data, limit_size=False):
        """
        Used to generate correct joined strings for multi-dataset case
        Specifically, handles "m4" data, since it doesn't require paths
        """
        names = [
            (data if ("m4" in path) or ("ran.csv" == path) else data + path)
            for path, data in zip(multi_data_path.split(","), multi_data.split(","))
        ]
        multi_dataset_name = "_".join(sorted(names))
        potential_hash = hashlib.shake_256(multi_dataset_name.encode()).hexdigest(
            12 // 2
        )
        return potential_hash if (limit_size) else multi_dataset_name

    @staticmethod
    def fetch_experiment_folders(
        given_experiment_folder: str,
        experiment_names: [str],
        use_strict_name_check=False,
        # clear_unnecessary_suffixes=["_cloudml"], # TODO: implement in the future, so that we can have _cloudml suffixes
    ):
        def check_name(exp: str, name: str, postfix: str = "_cloudml") -> bool:
            if use_strict_name_check:
                return exp == (name + "_cloudml")
            else:
                return exp.startswith(name)

        experiment_folders = []
        if (not Path(given_experiment_folder).is_dir()):
            print(f"Given experiment folder is empty: {given_experiment_folder}")
            return experiment_folders
        for model_checkpoint_folder in os.listdir(given_experiment_folder):
            # Generate full path
            full_model_checkpoint_folder_path = os.path.join(
                given_experiment_folder, model_checkpoint_folder
            )

            # Handles edge-case when the path is NOT a folder
            if not Path(full_model_checkpoint_folder_path).is_dir():
                continue

            # Check if folder belongs to specific experiment
            full_model_checkpoint_folder_experiments = os.listdir(
                full_model_checkpoint_folder_path
            )
            filtered_experiments = [
                os.path.join(full_model_checkpoint_folder_path, exp)
                for exp in full_model_checkpoint_folder_experiments
                # Checks if ANY of the experiment names start with given experiment names
                if any(check_name(exp, name) for name in experiment_names)
            ]
            experiment_folders.extend(filtered_experiments)
        return experiment_folders


def throw_noimpl(msg: Optional[str] = ""):
    raise NotImplementedError("Not implemented" if msg is None else msg)


def throw_invalid_state(msg: Optional[str] = ""):
    raise ValueError("Invalid state" if msg is None else msg)


def send_notification_email(
    subject: str,
    body: str,
    recipient_email_address: str,
    sender_email_address="my.email.notifier.02@gmail.com",
    sender_email_password="Slaptazodis",
    mail_smtp_server="smtp.gmail.com",
    mail_smtp_port=465,
):
    import smtplib
    import ssl
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    sender_email = sender_email_address
    receiver_email = recipient_email_address
    password = sender_email_password

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))
    body = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(mail_smtp_server, mail_smtp_port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, body)


# class PoolUtils:
#     import multiprocessing
#
#     @staticmethod
#     def starmap_with_kwargs(
#         pool: multiprocessing.Pool, fn, args_iter: list, kwargs_iter: list[dict]
#     ):
#         from itertools import repeat
#
#         def apply_args_and_kwargs(fn, args, kwargs):
#             return fn(*args, **kwargs)
#
#         args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
#         return pool.starmap(apply_args_and_kwargs, args_for_starmap)
