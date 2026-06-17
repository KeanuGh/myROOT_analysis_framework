from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import NotRequired, TypedDict

from src.cutting import Cut

NOMINAL_NAME = "T_s1thv_NOMINAL"
DTA_PATH = Path("/mnt/D/data/DTA_outputs/2024-09-19/")
DATA_PATH = Path("/data/DTA_outputs/2024-03-05/*data17*/*.root")
DSID_METADATA_CACHE = Path(__file__).with_name("dsid_meta_cache.json")

SelectionMap = Mapping[str, list[Cut]]
PathSpec = Path | list[Path] | dict[str, Path | list[Path]]
HardCutSpec = dict[str, str]


class SnapshotSpec(TypedDict):
    selections: SelectionMap | list[str]
    systematics: str


class Sample(TypedDict):
    data_path: PathSpec
    label: str
    hard_cut: NotRequired[HardCutSpec]
    is_data: NotRequired[bool]
    is_signal: NotRequired[bool]
    metadata_alias: NotRequired[str]
    selections: NotRequired[SelectionMap]
    snapshot: NotRequired[SnapshotSpec]


_DATA_SAMPLE: Sample = {
    "data_path": DATA_PATH,
    "label": "data",
    "is_data": True,
}

_MC_SAMPLES: dict[str, Sample] = {
    "wtaunu_had": {
        "data_path": {
            "lm_cut": DTA_PATH / "*Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
            "full": DTA_PATH / "*Sh_2211_Wtaunu_mW_120*/*.root",
        },
        "hard_cut": {
            "lm_cut": "(TruthBosonM < 120) && TruthTau_isHadronic",
            "full": "TruthTau_isHadronic",
        },
        "label": r"$W\rightarrow\tau\nu\rightarrow\mathrm{had}$",
        "is_signal": True,
        "metadata_alias": "wtaunu",
    },
    "wtaunu_lep": {
        "data_path": {
            "lm_cut": DTA_PATH / "*Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
            "full": DTA_PATH / "*Sh_2211_Wtaunu_mW_120*/*.root",
        },
        "hard_cut": {
            "lm_cut": "(TruthBosonM < 120) && !(TruthTau_isHadronic)",
            "full": "!(TruthTau_isHadronic)",
        },
        "label": r"$W\rightarrow\tau\nu\rightarrow\ell$",
        "metadata_alias": "wtaunu",
    },
    "wlnu": {
        "data_path": {
            "lm_cut": [
                DTA_PATH / "*Sh_2211_Wmunu_maxHTpTV2*/*.root",
                DTA_PATH / "*Sh_2211_Wenu_maxHTpTV2*/*.root",
            ],
            "full": [
                DTA_PATH / "*Sh_2211_Wmunu_mW_120*/*.root",
                DTA_PATH / "*Sh_2211_Wenu_mW_120*/*.root",
            ],
        },
        "hard_cut": {"lm_cut": "TruthBosonM < 120"},
        "label": r"$W\rightarrow (e/\mu)\nu$",
    },
    "zll": {
        "data_path": {
            "lm_cut": [
                DTA_PATH / "*Sh_2211_Ztautau_*_maxHTpTV2*/*.root",
                DTA_PATH / "*Sh_2211_Zee_maxHTpTV2*/*.root",
                DTA_PATH / "*Sh_2211_Zmumu_maxHTpTV2*/*.root",
            ],
            "full": [
                DTA_PATH / "*Sh_2211_Ztautau_mZ_120*/*.root",
                DTA_PATH / "*Sh_2211_Zmumu_mZ_120*/*.root",
                DTA_PATH / "*Sh_2211_Zee_mZ_120*/*.root",
                DTA_PATH / "*Sh_2211_Znunu_pTV2*/*.root",
            ],
        },
        "hard_cut": {"lm_cut": "TruthBosonM < 120"},
        "label": r"$Z\rightarrow (\ell/\nu)(\ell/\nu)$",
    },
    "top": {
        "data_path": [
            DTA_PATH / "*PP8_singletop*/*.root",
            DTA_PATH / "*PP8_tchan*/*.root",
            DTA_PATH / "*PP8_Wt_DR_dilepton*/*.root",
            DTA_PATH / "*PP8_ttbar_hdamp258p75*/*.root",
        ],
        "label": "Top",
    },
    "diboson": {
        "data_path": [
            DTA_PATH / "*Sh_2212_llll*/*.root",
            DTA_PATH / "*Sh_2212_lllv*/*.root",
            DTA_PATH / "*Sh_2212_llvv*/*.root",
            DTA_PATH / "*Sh_2212_lvvv*/*.root",
            DTA_PATH / "*Sh_2212_vvvv*/*.root",
            DTA_PATH / "*Sh_2211_ZqqZll*/*.root",
            DTA_PATH / "*Sh_2211_ZbbZll*/*.root",
            DTA_PATH / "*Sh_2211_WqqZll*/*.root",
            DTA_PATH / "*Sh_2211_WlvWqq*/*.root",
            DTA_PATH / "*Sh_2211_WlvZqq*/*.root",
            DTA_PATH / "*Sh_2211_WlvZbb*/*.root",
        ],
        "label": "Diboson",
    },
}


def _with_runtime_options(
    sample: Sample,
    *,
    selections: SelectionMap | None = None,
    snapshot: SnapshotSpec | None = None,
) -> Sample:
    sample = deepcopy(sample)
    if selections is not None:
        sample["selections"] = selections
    if snapshot is not None:
        sample["snapshot"] = snapshot
    return sample


def data_sample(selections: SelectionMap, *, snapshot: bool = False) -> Sample:
    return _with_runtime_options(
        _DATA_SAMPLE,
        selections=selections,
        snapshot={"selections": selections, "systematics": NOMINAL_NAME} if snapshot else None,
    )


def signal_sample(
    *,
    selections: SelectionMap | None = None,
    snapshot: SnapshotSpec | None = None,
) -> Sample:
    return _with_runtime_options(
        _MC_SAMPLES["wtaunu_had"], selections=selections, snapshot=snapshot
    )


def mc_samples(
    selections: SelectionMap,
    *,
    snapshot: bool = False,
) -> dict[str, Sample]:
    snapshot_config: SnapshotSpec | None = (
        {"selections": list(selections.keys()), "systematics": NOMINAL_NAME} if snapshot else None
    )
    return {
        name: _with_runtime_options(sample, selections=selections, snapshot=snapshot_config)
        for name, sample in _MC_SAMPLES.items()
    }


def analysis_samples(
    selections: SelectionMap,
    *,
    data_selections: SelectionMap | None = None,
    snapshot: bool = False,
) -> dict[str, Sample]:
    data_selections = selections if data_selections is None else data_selections
    return {
        "data": data_sample(data_selections, snapshot=snapshot),
        **mc_samples(selections, snapshot=snapshot),
    }
