import re
from glob import glob
from pathlib import Path
from warnings import warn


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


def smart_join(*s: str | None, sep: str = "_") -> str:
    """Concatenated string with separator, ignoring any blank strings or None"""
    return sep.join(filter(None, s))


def get_base_sys_name(s: str) -> str:
    """get the name of the base systematic"""
    return (
        # annoying typo: only JETID_EFF systematic doesn't have double underscore
        s.removeprefix("weight_")
        .removesuffix("_1up")
        .removesuffix("_1down")
        .removesuffix("_")
    )


def match_any(patterns: list | set | tuple, s: str) -> bool:
    """Match string s to any of the given patterns"""
    return any(re.match(p, s) for p in patterns)


def count_nested_values(d: dict) -> int:
    """Count number of subvalues in nested dict"""
    return sum([count_nested_values(v) if isinstance(v, dict) else 1 for v in d.values()])
