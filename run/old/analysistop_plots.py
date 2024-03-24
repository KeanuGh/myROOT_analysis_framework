import numpy as np

from analysis import Analysis

DTA_PATH = "/mnt/D/data/analysistop_out/mc16a/"
DATA_OUT_DIR = "/mnt/D/data/dataset_pkl_outputs/"
datasets = {
    "wmintaunu_analysistop": {
        "data_path": DTA_PATH + "wmintaunu_*/*.root",
        "cutfile": "../options/DTA_cuts/analysistop.txt",
        "label": r"wmintaunu",
    },
    "wmintaunu_peak_analysistop": {
        "data_path": DTA_PATH + "wmintaunu/*.root",
        "label": r"wmintaunu PEAK",
        "hard_cut": "M_W",
        "cutfile": "../options/DTA_cuts/analysistop_peak.txt",
    },
    "wplustaunu_analysistop": {
        "data_path": DTA_PATH + "wplustaunu_*/*.root",
        "cutfile": "../options/DTA_cuts/analysistop.txt",
        "label": r"wplustaunu",
    },
    "wplustaunu_peak_analysistop": {
        "data_path": DTA_PATH + "wplustaunu/*.root",
        "label": r"wplustaunu PEAK",
        "hard_cut": "M_W",
        "cutfile": "../options/DTA_cuts/analysistop_peak.txt",
    },
}
bins = np.array(
    [
        130,
        140.3921,
        151.6149,
        163.7349,
        176.8237,
        190.9588,
        206.2239,
        222.7093,
        240.5125,
        259.7389,
        280.5022,
        302.9253,
        327.1409,
        353.2922,
        381.5341,
        412.0336,
        444.9712,
        480.5419,
        518.956,
        560.4409,
        605.242,
        653.6246,
        705.8748,
        762.3018,
        823.2396,
        889.0486,
        960.1184,
        1036.869,
        1119.756,
        1209.268,
        1305.936,
        1410.332,
        1523.072,
        1644.825,
        1776.311,
        1918.308,
        2071.656,
        2237.263,
        2416.107,
        2609.249,
        2817.83,
        3043.085,
        3286.347,
        3549.055,
        3832.763,
        4139.151,
        4470.031,
        4827.361,
        5213.257,
    ]
)

my_analysis = Analysis(
    datasets,
    analysis_label="analysistop_analysis",
    # force_rebuild=True,
    TTree_name="truth",
    dataset_type="analysistop",
    log_level=10,
    lumi_year="2016+2015",
    data_dir=DATA_OUT_DIR,
    log_out="both",
    lepton="tau",
    # validate_duplicated_events=True,
    # force_recalc_weights=False,
)

my_analysis.merge_datasets("wmintaunu_analysistop", "wmintaunu_peak_analysistop")
my_analysis["wmintaunu_analysistop"]["truth_weight"] = (
    my_analysis["wmintaunu_analysistop"]["weight_mc"]
    * my_analysis["wmintaunu_analysistop"].lumi
    / my_analysis["wmintaunu_analysistop"]["weight_mc"].sum()
)

my_analysis.merge_datasets("wplustaunu_analysistop", "wplustaunu_peak_analysistop")
my_analysis["wplustaunu_analysistop"]["truth_weight"] = (
    my_analysis["wplustaunu_analysistop"]["weight_mc"]
    * my_analysis["wplustaunu_analysistop"].lumi
    / my_analysis["wplustaunu_analysistop"]["weight_mc"].sum()
)

my_analysis.plot_hist(
    "wmintaunu_analysistop",
    "MC_WZ_dilep_m_born",
    bins=bins,
    weight="truth_weight",
    logx=True,
    stats_box=True,
)
my_analysis.plot_hist(
    "wmintaunu_analysistop",
    "MC_WZ_dilep_m_born",
    bins=bins,
    logx=True,
    stats_box=True,
    suffix="unweighted",
)

my_analysis.logger.info("DONE")
