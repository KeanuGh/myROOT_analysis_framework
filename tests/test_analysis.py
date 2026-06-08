import json
import shutil

import numpy as np
import pytest
import ROOT

from src.analysis import Analysis
from src.cutting import Cut
from src.dataset import Dataset


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


class TestBinByBinUnfolding:
    def test_bin_by_bin_correction_propagates_uncertainty(self):
        reco = ROOT.TH1F("bbb_reco", "bbb_reco", 1, 0, 1)
        truth = ROOT.TH1F("bbb_truth", "bbb_truth", 1, 0, 1)
        measured = ROOT.TH1F("bbb_measured", "bbb_measured", 1, 0, 1)

        reco.SetBinContent(1, 50)
        reco.SetBinError(1, 5)
        truth.SetBinContent(1, 100)
        truth.SetBinError(1, 10)
        measured.SetBinContent(1, 25)
        measured.SetBinError(1, 3)

        correction = Analysis.get_bin_by_bin_correction(reco, truth)
        unfolded = Analysis.apply_bin_by_bin_correction(measured, correction)

        expected_correction = 2.0
        expected_correction_error = np.sqrt((10 / 50) ** 2 + (100 * 5 / 50 ** 2) ** 2)
        expected_unfolded = 50.0
        expected_unfolded_error = np.sqrt(
            (3 * expected_correction) ** 2 + (25 * expected_correction_error) ** 2
        )

        np.testing.assert_allclose(correction.GetBinContent(1), expected_correction)
        np.testing.assert_allclose(correction.GetBinError(1), expected_correction_error)
        np.testing.assert_allclose(unfolded.GetBinContent(1), expected_unfolded)
        np.testing.assert_allclose(unfolded.GetBinError(1), expected_unfolded_error)


class TestSystematicUncertainties:
    @staticmethod
    def make_hist(name: str, value: float) -> ROOT.TH1F:
        hist = ROOT.TH1F(name, name, 1, 0, 1)
        hist.SetDirectory(0)
        hist.SetBinContent(1, value)
        return hist

    def test_systematic_uncertainty_uses_max_deviation_from_nominal(self):
        dataset = Dataset(
            name="test",
            rdataframes={"T_s1thv_NOMINAL": ROOT.RDataFrame(1)},
        )
        dataset.nominal_name = "T_s1thv_NOMINAL"
        dataset.tes_sys_set = {"SME_TES_TEST__1up", "SME_TES_TEST__1down"}
        dataset.histograms = {
            "T_s1thv_NOMINAL": {"sel": {"TauPhi": self.make_hist("nominal", 100)}},
            "SME_TES_TEST__1up": {"sel": {"TauPhi": self.make_hist("up", 110)}},
            "SME_TES_TEST__1down": {"sel": {"TauPhi": self.make_hist("down", -90)}},
        }

        dataset.calculate_systematic_uncertainties()

        total = dataset.histograms["T_s1thv_NOMINAL"]["sel"]["TauPhi_SME_TES_TEST_tot_uncert"]
        percent = dataset.histograms["T_s1thv_NOMINAL"]["sel"]["TauPhi_SME_TES_TEST_pct_uncert"]

        np.testing.assert_allclose(total.GetBinContent(1), 190)
        np.testing.assert_allclose(percent.GetBinContent(1), 190)

    def test_dataset_systematic_uncertainty_combines_sources_in_quadrature(self):
        dataset = Dataset(
            name="test",
            rdataframes={"T_s1thv_NOMINAL": ROOT.RDataFrame(1)},
        )
        dataset.nominal_name = "T_s1thv_NOMINAL"
        dataset.tes_sys_set = {
            "SME_TES_A__1up",
            "SME_TES_A__1down",
            "SME_TES_B__1up",
            "SME_TES_B__1down",
        }
        dataset.histograms = {
            "T_s1thv_NOMINAL": {
                "sel": {
                    "TauPhi": self.make_hist("nominal", 100),
                    "TauPhi_SME_TES_A_tot_uncert": self.make_hist("a", 3),
                    "TauPhi_SME_TES_B_tot_uncert": self.make_hist("b", 4),
                }
            }
        }

        down, up = dataset.get_systematic_uncertainty("TauPhi", "sel")

        np.testing.assert_allclose(down, [5])
        np.testing.assert_allclose(up, [5])
