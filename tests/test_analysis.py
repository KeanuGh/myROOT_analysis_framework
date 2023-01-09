from analysis import Analysis


class TestSimpleAnalysis:
    datasets = {
        "wmintaunu": {
            "data_path": "tests/resources/test_analysistop_mcwmintaunu.root",
            "cutfile_path": "tests/resources/cutfile_EXAMPLE.txt",
            "TTree_name": "truth",
            "hard_cut": r"Muon $|#eta|$",
            "lepton": "tau",
            "label": r"$W^-\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
    }

    analysis = Analysis(
        datasets,
        analysis_label="test_analysis",
        force_rebuild=True,
        dataset_type="analysistop",
        log_level=10,
        log_out="console",
        year="2015+2016",
    )

    def test_n_events(self):
        assert len(self.analysis["wmintaunu"]) == 331000

    # def test_cutflow(self):
    #     cutflow = self.analysis['wmintaunu'].cutflow
    #
    #     assert cutflow.cutflow_n_events == 331000
    #     assert cutflow.cu
