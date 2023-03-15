import shutil

import pandas as pd  # type: ignore
import pytest

from analysis import Analysis


class TestSimpleAnalysisTop:
    @pytest.fixture(scope="class")
    def analysis(self) -> Analysis:
        output_dir = "tests/framework_test_outputs"
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
            "tests/framework_test_outputs/outputs/test_analysis/pickles/wmintaunu_df.pkl"
        )
        assert pkl_output.equals(analysis["wmintaunu"].df)


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
