import os
from glob import glob
from filecmp import cmp
from typing import Optional, Union, List, Generator
from pathlib import Path
import analysis.config as config
from warnings import warn
from utils import ROOT_utils


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


def get_filename(filepath: str, suffix: bool = False) -> str:
    """gets the name of file contained in path"""
    if suffix:
        return Path(filepath).name
    else:
        return Path(filepath).stem


def get_file_parent(filepath: str) -> str:
    """Get full path of directory containing file"""
    return Path(filepath).parent


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


def convert_pkl_to_root(pkl_name: Optional[str] = None, conv_all: bool = False) -> None:
    """
    Converts histogram pickle file into a root file containing the same histograms.
    At the moment only converts data and doesn't touch title and axis labels but may change that in the future
     ^ (written 12 Nov 2020)
    """
    if pkl_name:
        assert file_exists(pkl_name), FileNotFoundError(f"Could not find pickle file {pkl_name}")
        if get_file_parent(pkl_name) in (config.pkl_hist_dir, '.') \
                and conv_all:
            warn("Given file is contained in pickle file directory. Will convert all.")
        else:
            # do just that file in pkl_hist_dir
            for file in glob(pkl_name):
                ROOT_utils.convert_pkl_to_root(file)
    if conv_all:
        # conv all
        for file in glob(config.pkl_hist_dir + '*'):
            if not file.endswith('.pkl'):
                continue
            ROOT_utils.convert_pkl_to_root(file)
    else:
        raise ValueError("Choose pkl histogram file(s) to convert or choose to convert all for this analysis")
