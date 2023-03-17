import shutil

import numpy as np
import pandas as pd  # type: ignore
import pytest

from analysis import Analysis


class TestSimpleAnalysisTop:
    @pytest.fixture(scope="class")
    def analysis(self) -> Analysis:
        output_dir = "framework_test_outputs"
        datasets = {
            "wmintaunu": {
                "data_path": "resources/test_analysistop_mcwmintaunu.root",
                "cutfile": "resources/cutfile_eg_analysistop.txt",
                "TTree_name": "truth",
                "lepton": "tau",
                "label": r"$W^-\rightarrow\tau\nu\rightarrow\mu\nu$",
            },
        }

        yield Analysis(
            datasets,
            analysis_label="test_analysis",
            output_dir=output_dir,
            force_rebuild=True,
            dataset_type="analysistop",
            log_level=10,
            log_out="console",
            year="2015+2016",
        )

        # delete outputs
        shutil.rmtree(output_dir)

    def test_n_events(self, analysis):
        assert len(analysis["wmintaunu"]) == 331000

    def test_truth_events(self, analysis):
        assert analysis["wmintaunu"].n_truth_events == 331000

    def test_reco_events(self, analysis):
        assert analysis["wmintaunu"].n_reco_events == 84

    def test_cutflow(self, analysis):
        cutflow = analysis["wmintaunu"].cutflow

        assert cutflow.cutflow_n_events[0] == 331000
        assert cutflow.cutflow_n_events[1] == 202149
        assert cutflow.cutflow_n_events[2] == 153558
        assert cutflow.cutflow_n_events[3] == 130104

    def test_pickle(self, analysis):
        pkl_output = pd.read_pickle(
            "framework_test_outputs/outputs/test_analysis/pickles/wmintaunu_df.pkl"
        )
        assert pkl_output.equals(analysis["wmintaunu"].df)

    def test_histograms(self, analysis):
        histograms = analysis["wmintaunu"].gen_histograms(cut=True)

        assert histograms["MC_WZmu_el_eta_born"].n_bins == 30
        assert round(histograms["mu_pt"].bin_values()[0], 5) == 283.18032


class TestSimpleDTA:
    @pytest.fixture(scope="class")
    def analysis(self) -> Analysis:
        output_dir = "tests/framework_test_outputs"
        datasets = {
            "wmintaunu": {
                "data_path": "resources/wtaunu_h_cvbv_1000.root",
                "cutfile": "resources/cutfile_eg_dta.txt",
                "TTree_name": "T_s1thv_NOMINAL",
                "lepton": "tau",
                "label": r"$W^-\rightarrow\tau\nu\rightarrow\mu\nu$",
            },
        }

        yield Analysis(
            datasets,
            analysis_label="test_analysis",
            output_dir=output_dir,
            force_rebuild=True,
            dataset_type="dta",
            log_level=10,
            log_out="console",
            year="2015+2016",
        )

        # delete outputs
        shutil.rmtree(output_dir)

    def test_n_events(self, analysis):
        analysis["wmintaunu"].gen_cutflow()
        assert len(analysis["wmintaunu"]) == 5000

    def test_cutflow(self, analysis):
        analysis["wmintaunu"].apply_cuts()

        assert analysis["wmintaunu"].cutflow["tau_eta"].value == "abs(TruthTauEta) < 2.4"
        assert analysis["wmintaunu"].cutflow["tau_eta"].npass == 3577

    def test_histograms(self, analysis):
        histograms = analysis["wmintaunu"].gen_histograms(cut=True)

        assert histograms["TauEta"].n_bins == 30

        np.testing.assert_allclose(
            histograms["TruthTauPt"].bin_values(),
            np.array(
                [
                    4.02496986e01,
                    3.50997391e01,
                    8.88594116e02,
                    7.75344299e02,
                    3.19974060e02,
                    1.51049792e03,
                    4.37209229e03,
                    6.09370801e03,
                    1.28474688e04,
                    2.77360547e04,
                    4.21289922e04,
                    8.46178984e04,
                    1.37219031e05,
                    6.33462969e04,
                    9.65051074e03,
                    2.51681860e03,
                    9.62219299e02,
                    2.85374146e02,
                    9.01328125e01,
                    7.12713575e00,
                    1.04047089e01,
                    4.74639368e00,
                    9.92996693e-01,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ]
            ),
        )
