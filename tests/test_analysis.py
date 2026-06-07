import json
import shutil

import numpy as np
import pytest

from src.analysis import Analysis
from src.cutting import Cut


@pytest.fixture(scope="class")
def dta_analysis(tmp_directory) -> Analysis:
    output_dir = tmp_directory / "framework_test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dsid_meta_cache.json", "w") as f:
        json.dump(
            {
                "metadata": {
                    "700346": {
                        "dsid": 700346,
                        "sumw": 503981998604288.0,
                        "cross_section": 1.0,
                        "kfactor": 1.0,
                        "filter_eff": 1.0,
                        "phys_short": "test_wtaunu",
                        "generator_name": "test",
                        "total_events": 5000,
                        "total_size": "",
                        "etag": "",
                        "ptag": "",
                        "stag": "",
                        "rtag": "",
                    }
                },
                "dataset_ids": {"wmintaunu": [700346]},
            },
            f,
        )

    datasets = {
        "wmintaunu": {
            "data_path": "tests/resources/wtaunu_h_cvbv_1000.root",
            "label": r"$W^-\\rightarrow\\tau\\nu$",
            "selections": {
                "fiducial_truth": [
                    Cut("met", "MET_met > 150"),
                    Cut("met phi", "MET_phi > 0"),
                ]
            },
        },
    }

    yield Analysis(
        datasets,
        analysis_label="test_analysis",
        output_dir=output_dir,
        rerun=True,
        regen_histograms=True,
        ttree="T_s1thv_NOMINAL",
        log_level=30,
        log_out="console",
        year=2017,
        extract_vars={"MET_met", "MET_phi"},
        binnings={"": {}},
    )

    shutil.rmtree(output_dir)


class TestSimpleDTA:
    def test_dataset_is_loaded(self, dta_analysis):
        assert dta_analysis["wmintaunu"].rdataframes["T_s1thv_NOMINAL"].Count().GetValue() == 5000
        assert dta_analysis.mc_samples == ["wmintaunu"]

    def test_cutflow(self, dta_analysis):
        cutflow = dta_analysis["wmintaunu"].cutflows["T_s1thv_NOMINAL"]["fiducial_truth"]
        assert [item.cut.name for item in cutflow] == ["met", "met phi"]
        assert cutflow[0].npass == 389
        assert cutflow[1].npass == 175

    def test_histograms(self, dta_analysis):
        histograms = dta_analysis["wmintaunu"].histograms["T_s1thv_NOMINAL"]["fiducial_truth"]
        assert histograms["MET_met"].GetNbinsX() == 30
        assert histograms["MET_phi"].GetNbinsX() == 30
        assert histograms["MET_met"].Integral() > 0

    def test_histogram_round_trip(self, dta_analysis, tmp_directory):
        path = tmp_directory / "dta_histograms.root"
        original = dta_analysis["wmintaunu"].histograms["T_s1thv_NOMINAL"]["fiducial_truth"][
            "MET_met"
        ].Clone()

        dta_analysis["wmintaunu"].export_histograms(path)
        dta_analysis["wmintaunu"].histograms = {}
        dta_analysis["wmintaunu"].import_dataset(path)

        restored = dta_analysis["wmintaunu"].histograms["T_s1thv_NOMINAL"]["fiducial_truth"][
            "MET_met"
        ]
        np.testing.assert_allclose(
            [restored.GetBinContent(i + 1) for i in range(restored.GetNbinsX())],
            [original.GetBinContent(i + 1) for i in range(original.GetNbinsX())],
        )
