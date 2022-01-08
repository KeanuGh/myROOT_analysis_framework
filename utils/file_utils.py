import logging
import os
from filecmp import cmp
from glob import glob
from pathlib import Path
from typing import Optional, Union, List
from warnings import warn

import src.config as config
from utils import ROOT_utils


def get_last_backup(backup_dir: str, name: str = '') -> Optional[str]:
    if is_dir_empty(backup_dir) or len(glob(backup_dir + name + '*')) == 0:
        return None
    else:
        return max(glob(backup_dir + name + '*'), key=os.path.getctime)


def identical_to_backup(file: str,
                        logger: logging.Logger,
                        backup_dir: Optional[str] = None,
                        backup_file: Optional[str] = None,
                        name: str = '',
                        ) -> bool:
    """
    checks whether current file is the same as the last backup file.
    input either backup directory or backup file.
    Returns False if backup_file or backup_dir is None
    """
    if backup_dir and backup_file:
        raise Exception("Input either directory or filepath")
    elif not backup_dir and not backup_file:
        logger.warning("Backup directory is empty")
        return False
    elif backup_dir:
        backup_file = get_last_backup(backup_dir, name)
    return cmp(file, backup_file)


def is_dir_empty(dirpath: str) -> bool:
    """checks if input directory is empty"""
    return len(os.listdir(dirpath)) == 0


def delete_file(file: str) -> None:
    """deletes a file"""
    try:
        os.remove(file)
    except FileNotFoundError:
        warn(f"No file named {file}. No file deleted")


def get_filename(filepath: str, suffix: bool = False) -> str:
    """gets the name of file contained in path"""
    if suffix:
        return Path(filepath).name
    else:
        return Path(filepath).stem


def get_file_parent(filepath: str) -> Path:
    """Get full path of directory containing file"""
    return Path(filepath).parent


def makedir(dirpath: Union[str, List[str]]) -> None:
    """creates director(y/ies) if it/they don't exist. Accepts either string or list of strings"""
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
    return bool(glob(filepath))


def n_files(filepath: str) -> int:
    """how many files?"""
    return len(glob(filepath))


# FIXME
def convert_pkl_to_root(pkl_name: Optional[str] = None, conv_all: bool = False) -> None:
    """
    Converts histogram pickle file into a root file containing the same histograms.
    At the moment only converts data and doesn't touch title and axis labels but may change that in the future
     ^ (written 12 Nov 2020)
    """
    if pkl_name:
        if not file_exists(pkl_name):
            raise FileNotFoundError(f"Could not find pickle file {pkl_name}")
        if get_file_parent(pkl_name) in (config.paths['pkl_hist_dir'], '.') \
                and conv_all:
            warn("Given file is contained in pickle file directory. Will convert all.")
        else:
            # do just that file in pkl_hist_dir
            for file in glob(pkl_name):
                ROOT_utils.convert_pkl_to_root(file)
    if conv_all:
        # conv all
        for file in glob(config.paths['pkl_hist_dir'] + '*'):
            if not file.endswith('.pkl'):
                continue
            ROOT_utils.convert_pkl_to_root(file)
    else:
        raise ValueError("Choose pkl histogram file(s) to convert or choose to convert all for this src")
