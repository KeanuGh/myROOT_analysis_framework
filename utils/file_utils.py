import os
from glob import glob
from pathlib import Path
from warnings import warn


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


def multi_glob(paths: Path | list[Path] | str | list[str]) -> list[str]:
    """Return list of files from list of paths with wildcards"""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    all_files = []
    for path in paths:
        f = glob(str(path))
        if not f:
            warn(f"Path passed with no files: {path}")

        all_files += f

    return all_files


# didn't know where else to put this
def smart_join(s: list[str | None], sep: str = "_") -> str:
    """Concatenated string with separator, ignoring any blank strings or None"""
    return sep.join(filter(None, s))
