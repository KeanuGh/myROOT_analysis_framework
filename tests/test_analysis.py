import shutil

import numpy as np
import pandas as pd  # type: ignore
import pytest

from analysis import Analysis
from cutflow import cutflow_from_hist_and_cutfile


class TestSimpleAnalysisTop:
    @pytest.fixture(scope="class")
    def analysis(self, tmp_directory) -> Analysis:
        output_dir = tmp_directory / "framework_test_outputs"
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

    def test_pickle(self, analysis, tmp_directory):
        pkl_output = pd.read_pickle(
            tmp_directory / "framework_test_outputs/outputs/test_analysis/pickles/wmintaunu_df.pkl"
        )
        assert pkl_output.equals(analysis["wmintaunu"].df)

    def test_histograms(self, analysis, tmp_directory):
        histograms = analysis["wmintaunu"].histograms

        assert histograms["MC_WZmu_el_eta_born"].GetNbinsX() == 30
        np.testing.assert_allclose(histograms["mu_pt"].GetBinContent(15), 17323.02344, rtol=1e-6)

        # save histograms to file
        path = tmp_directory / "analysistop_histograms.root"
        analysis["wmintaunu"].save_hists_to_file(path)

        # read
        output_hists = analysis["wmintaunu"].import_histograms(path, inplace=False)

        np.testing.assert_allclose(
            [
                output_hists["MC_WZmu_el_eta_born"].GetBinContent(i + 1)
                for i in range(output_hists["MC_WZmu_el_eta_born"].GetNbinsX())
            ],
            [
                histograms["MC_WZmu_el_eta_born"].GetBinContent(i + 1)
                for i in range(histograms["MC_WZmu_el_eta_born"].GetNbinsX())
            ],
            rtol=1e-5,
        )


class TestSimpleDTA:
    @pytest.fixture(scope="class")
    def analysis(self, tmp_directory) -> Analysis:
        output_dir = tmp_directory / "tests/framework_test_outputs"
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

    # @pytest.fixture(scope="class")
    # def tau_pt_histogram(self) -> ROOT.TH1F:
    #

    def test_n_events(self, analysis):
        assert len(analysis["wmintaunu"]) == 5000

    def test_cutflow(self, analysis):
        assert analysis["wmintaunu"].cutflow[1].cut.cutstr == "abs(TruthTauEta) < 2.4"
        assert analysis["wmintaunu"].cutflow[1].npass == 3577

    def test_histograms(self, analysis, tmp_directory):
        histograms = analysis["wmintaunu"].histograms

        assert histograms["TauEta"].GetNbinsX() == 30

        np.testing.assert_allclose(
            np.array(
                [
                    histograms["TruthTauPt"].GetBinContent(i + 1)
                    for i in range(histograms["TruthTauPt"].GetNbinsX())
                ]
            ),
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

        # save histograms to file
        path = tmp_directory / "dta_histograms.root"
        analysis["wmintaunu"].save_hists_to_file(path)

        # read
        output_hists = analysis["wmintaunu"].import_histograms(path, inplace=False)

        np.testing.assert_allclose(
            [
                output_hists["TruthTauPt"].GetBinContent(i + 1)
                for i in range(output_hists["TruthTauPt"].GetNbinsX())
            ],
            [
                histograms["TruthTauPt"].GetBinContent(i + 1)
                for i in range(histograms["TruthTauPt"].GetNbinsX())
            ],
        )

    def test_cutflow_self_consistency(self, analysis, tmp_directory):
        generated_cutflow = analysis["wmintaunu"].cutflow._cutflow

        regenerated_cutflow = cutflow_from_hist_and_cutfile(
            analysis["wmintaunu"].histograms["cutflow"], analysis["wmintaunu"].cutfile
        )

        assert [item.cut.cutstr for item in generated_cutflow] == [
            item.cut.cutstr for item in regenerated_cutflow
        ]

        for generated_cutflow_item, regenerated_cutflow_item in zip(
            generated_cutflow, regenerated_cutflow
        ):
            assert np.isclose(generated_cutflow_item.npass, regenerated_cutflow_item.npass)
            assert np.isclose(generated_cutflow_item.eff, regenerated_cutflow_item.eff)
            assert np.isclose(generated_cutflow_item.ceff, regenerated_cutflow_item.ceff)
