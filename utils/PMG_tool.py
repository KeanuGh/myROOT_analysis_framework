from pathlib import Path

import numpy as np
import pandas as pd

PMG_DB_FILE = "PMGxsecDB_mc16.txt"

try:
    PMG_df = pd.read_csv(
        Path(__file__).parent / PMG_DB_FILE,
        delim_whitespace=True,
        header=0,
        index_col=0,
        names=[
            "DSID",
            "physics_short",
            "crossSection",
            "genFiltEff",
            "kFactor",
            "relUncertUP",
            "relUncertDOWN",
            "generator_name",
            "etag",
        ],
        dtype={
            "DSID": np.int32,
            "physics_short": str,
            "crossSection": np.float32,
            "genFiltEff": np.float32,
            "kFactor": np.float32,
            "relUncertUP": np.float32,
            "relUncertDOWN": np.float32,
            "generator_name": str,
            "etag": str,
        },
    )
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find PMG database file '{PMG_DB_FILE}'")


def get_crossSection(dsid: int) -> float:
    return PMG_df.loc[dsid, "crossSection"]


def get_physics_short(dsid: int) -> str:
    return PMG_df.loc[dsid, "physics_short"]


def get_genFiltEff(dsid: int) -> float:
    return PMG_df.loc[dsid, "genFiltEff"]


def get_kFactor(dsid: int) -> float:
    return PMG_df.loc[dsid, "kFactor"]


def get_relUncertUP(dsid: int) -> float:
    return PMG_df.loc[dsid, "relUncertUP"]


def get_relUncertDOWN(dsid: int) -> float:
    return PMG_df.loc[dsid, "relUncertDOWN"]


def get_generator_name(dsid: int) -> str:
    return PMG_df[dsid, "generator_name"]


def get_etag(dsid: int) -> str:
    return PMG_df[dsid, "etag"]
