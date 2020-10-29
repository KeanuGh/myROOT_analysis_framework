import os
from glob import glob
from filecmp import cmp
from typing import Optional
from pathlib import Path


def get_last_backup(backup_dir: str) -> str:
    return max(glob(backup_dir + '*'), key=os.path.getctime)


def identical_to_backup(file: str,
                        backup_dir: Optional[str] = None,
                        backup_file: Optional[str] = None
                        ) -> bool:
    """
    checks whether current file is the same as the last backup file.
    input either backup directory or backup file
    """
    if (backup_dir and backup_file) or (not backup_dir and not backup_file):
        raise Exception("Input either directory or filepath")
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


def makedir(dirpath: str) -> None:
    """creates directory if it doesn't exist"""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
