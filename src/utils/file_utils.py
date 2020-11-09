import os
from glob import glob
from filecmp import cmp
from typing import Optional, Union, List
from pathlib import Path
import analysis.config as config
from warnings import warn


def get_last_backup(backup_dir: str) -> Optional[str]:
    if is_dir_empty(backup_dir):
        return None
    else:
        return max(glob(backup_dir + '*'), key=os.path.getctime)


def identical_to_backup(file: str, backup_dir: Optional[str] = None, backup_file: Optional[str] = None) -> bool:
    """
    checks whether current file is the same as the last backup file.
    input either backup directory or backup file.
    Returns False if backup_file or backup_dir is None
    """
    if backup_dir and backup_file:
        raise Exception("Input either directory or filepath")
    elif not backup_dir and not backup_file:
        warn("Backup directory is empty")
        return False
    elif backup_dir:
        backup_file = get_last_backup(backup_dir)
    return cmp(file, backup_file)


def is_dir_empty(dirpath: str) -> bool:
    """checks if input directory is empty"""
    return len(os.listdir(dirpath)) == 0


def delete_file(file: str) -> None:
    """deletes a file"""
    os.remove(file)


def get_filename(filepath: str) -> str:
    """gets the name of file contained in path"""
    return Path(filepath).name


def makedir(dirpath: Union[str, List[str]]) -> None:
    """creates director(y/ies) if it/they doesn't exist. Accepts either string or list of strings"""
    if isinstance(dirpath, str):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    elif isinstance(dirpath, list):
        for path in dirpath:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        raise ValueError("dirpath should be a string or a list of strings")


def file_exists(filepath: str) -> bool:
    """Does the file exist?"""
    return os.path.isfile(filepath)


def clear_pkl(ds_name: Optional[Union[List[str], str]] = None, clear_all: bool = False) -> None:
    """
    Deletes given pkl dataset. If clear_all set as True, clears all pkl_datasets

    :param clear_all: whether to clear all files in dataset pickle dir
    :param ds_name: OPTIONAL. String or list of string name(s) of dataset(s) to remove
    """
    path = config.pkl_df_filepath
    assert ds_name or clear_all, "Must give a dataset name or clear_all"

    assert not is_dir_empty(path.rstrip('{}_df.pkl')), "Pickle directory empty"
    if clear_all:
        delete_file(path.format('*'))
        return
    elif isinstance(ds_name, list):
        for name in ds_name:
            assert isinstance(name, str), "Dataset name must be a string"
            assert file_exists(path.format(name)), f"{path.format(name)} does not exist."
            delete_file(path.format(name))
    elif isinstance(ds_name, str):
        assert file_exists(path.format(ds_name)), f"{path.format(ds_name)} does not exist."
        delete_file(path.format(ds_name))
    raise ValueError(f"ds_name should supply string or list of string name(s) of dataset(s) to remove")

