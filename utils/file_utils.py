import os
from glob import glob
from pathlib import Path


def is_dir_empty(dirpath: str) -> bool:
    """checks if input directory is empty"""
    return len(os.listdir(dirpath)) == 0


def get_filename(filepath: str | Path, suffix: bool = False) -> str:
    """gets the name of file contained in path"""
    if suffix:
        return Path(filepath).name
    else:
        return Path(filepath).stem


def get_file_parent(filepath: str) -> Path:
    """Get full path of directory containing file"""
    return Path(filepath).parent


def file_exists(filepath: str | Path) -> bool:
    """Does the file exist?"""
    if isinstance(filepath, Path):
        return bool(filepath.parent.rglob("*"))
    return bool(glob(filepath))


def n_files(filepath: str | Path) -> int:
    """how many files?"""
    if isinstance(filepath, Path):
        return len(list(filepath.parent.rglob("*.root")))
    return len(glob(filepath))


# # FIXME
# def convert_pkl_to_root(pkl_name: str | None = None, conv_all: bool = False) -> None:
#     """
#     Converts histogram pickle file into a root file containing the same histograms.
#     At the moment only converts data and doesn't touch title and axis labels but may change that in the future
#      ^ (written 12 Nov 2020)
#     """
#     if pkl_name:
#         if not file_exists(pkl_name):
#             raise FileNotFoundError(f"Could not find pickle file {pkl_name}")
#         if get_file_parent(pkl_name) in (config.paths["pkl_hist_dir"], ".") and conv_all:
#             warn("Given file is contained in pickle file directory. Will convert all.")
#         else:
#             # do just that file in pkl_hist_dir
#             for file in glob(pkl_name):
#                 ROOT_utils.convert_pkl_to_root(file)
#     if conv_all:
#         # conv all
#         for file in glob(config.paths["pkl_hist_dir"] + "*"):
#             if not file.endswith(".pkl"):
#                 continue
#             ROOT_utils.convert_pkl_to_root(file)
#     else:
#         raise ValueError(
#             "Choose pkl histogram file(s) to convert or choose to convert all for this src"
#         )
