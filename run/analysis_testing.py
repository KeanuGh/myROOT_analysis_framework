from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset
from src.analysis import Analysis
from src.cutting import Cut
from utils import ROOT_utils
from utils.variable_names import variable_data

DTA_PATH = Path("/mnt/D/data/DTA_outputs/2024-08-28/")
NOMINAL_NAME = "T_s1thv_NOMINAL"

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        # SIGNAL
        # ====================================================================
        "wtaunu": {
            "data_path": "/mnt/D/data/DTA_outputs/2024-09-19/user.kghorban.Sh_2211_Wtaunu_H_maxHTpTV2_BFilter.e8351.MC16d.v1.2024-09-19_histograms.root/user.kghorban.41345350._000001.histograms.root",
            "label": r"$W\rightarrow\tau\nu$",
            "is_signal": True,
        },
    }

    # CUTS & SELECTIONS
    # ========================================================================
    pass_presel = Cut(
        r"Pass preselection",
        r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1)"
        r"&& ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)",
    )
    pass_taupt170 = Cut(
        r"$p_T^\tau > 170$",
        r"TauPt > 170",
    )
    pass_mtw150 = Cut(
        r"$m_T^W > 150$",
        r"MTW > 150",
    )
    pass_loose = Cut(
        r"\mathrm{Pass Loose ID}",
        r"(TauBDTEleScore > 0.05) && "
        r"((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
    )
    fail_loose = Cut(
        r"\mathrm{Fail Loose ID}",
        "(TauBDTEleScore > 0.05) && "
        "!((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
    )
    pass_met150 = Cut(
        r"$E_T^{\mathrm{miss}} > 150$",
        r"MET_met > 150",
    )
    pass_100met = Cut(
        r"$E_T^{\mathrm{miss}} < 100$",
        r"MET_met < 100",
    )
    # this is a multijet background estimation. Tau is "True" in this case if it is a lepton
    pass_truetau = Cut(
        r"True Tau",
        "MatchedTruthParticle_isHadronicTau == true || MatchedTruthParticle_isMuon == true || MatchedTruthParticle_isElectron == true",
    )

    # selections
    selections: dict[str, list[Cut]] = {
        "SR_passID": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            pass_loose,
            pass_met150,
        ],
    }

    wanted_variables = {"TauPt", "MTW", "eventNumber"}

    # RUN
    # ========================================================================
    analysis = Analysis(
        datasets,
        year=2017,
        # rerun=True,
        # regen_histograms=True,
        do_systematics=True,
        # regen_metadata=True,
        ttree="T_s1thv_NOMINAL",
        selections=selections,
        analysis_label="analysis_test",
        dataset_type="dta",
        log_level=10,
        log_out="both",
        extract_vars=wanted_variables,
        import_missing_columns_as_nan=True,
        binnings={
            "": {
                "TauPt": np.geomspace(170, 1000, 21),
            },
        },
    )
    print("TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt:")
    hist_sys_down = analysis["wtaunu"].get_hist(
        "TauPt", "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt__1down", "SR_passID"
    )
    hist_nom = analysis["wtaunu"].get_hist("TauPt", "T_s1thv_NOMINAL", "SR_passID")
    hist_sys_up = analysis["wtaunu"].get_hist(
        "TauPt", "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt__1up", "SR_passID"
    )

    print("sys_down: ", ROOT_utils.get_th1_bin_values(hist_sys_down))
    print("nominal: ", ROOT_utils.get_th1_bin_values(hist_nom))
    print("sys_up: ", ROOT_utils.get_th1_bin_values(hist_sys_up))

    print("TAUS_TRUEHADTAU_EFF_RECO_TOTAL:")
    hist_sys_down = analysis["wtaunu"].get_hist(
        "TauPt", "TAUS_TRUEHADTAU_EFF_RECO_TOTAL__1down", "SR_passID"
    )
    hist_nom = analysis["wtaunu"].get_hist("TauPt", "T_s1thv_NOMINAL", "SR_passID")
    hist_sys_up = analysis["wtaunu"].get_hist(
        "TauPt", "TAUS_TRUEHADTAU_EFF_RECO_TOTAL__1up", "SR_passID"
    )
    hist_sys_tot = analysis["wtaunu"].get_hist(
        "TauPt_TAUS_TRUEHADTAU_EFF_RECO_TOTAL_tot_uncert", NOMINAL_NAME, "SR_passID"
    )
    hist_sys_pct = analysis["wtaunu"].get_hist(
        "TauPt_TAUS_TRUEHADTAU_EFF_RECO_TOTAL_pct_uncert", NOMINAL_NAME, "SR_passID"
    )

    print("sys_down: ", ROOT_utils.get_th1_bin_values(hist_sys_down))
    print("nominal: ", ROOT_utils.get_th1_bin_values(hist_nom))
    print("sys_up: ", ROOT_utils.get_th1_bin_values(hist_sys_up))
    print("tot_uncert: ", ROOT_utils.get_th1_bin_values(hist_sys_tot))
    print("pct_uncert: ", ROOT_utils.get_th1_bin_values(hist_sys_pct))

    # try summing manually
    sys_up_sum = analysis["wtaunu"].get_hist(
        f"TauPt_TAUS_TRUEHADTAU_EFF_RECO_TOTAL__1up_diff", NOMINAL_NAME, "SR_passID"
    )
    for sys in analysis["wtaunu"].eff_sys_set | analysis["wtaunu"].tes_sys_set:
        if "TAUS_TRUEHADTAU_EFF_RECO_TOTAL" in sys:
            continue
        # if "EFF" in sys:
        #     continue
        if sys.endswith("_1down"):
            continue
        sys_up_sum += analysis["wtaunu"].get_hist(f"TauPt_{sys}_diff", NOMINAL_NAME, "SR_passID")

    print("sys_up_sum: ", ROOT_utils.get_th1_bin_values(sys_up_sum))

    analysis.plot(
        val="TauPt",
        dataset="wtaunu",
        selection="SR_passID",
        xlabel=variable_data["TauPt"]["name"] + " [GeV]",
        ylabel="Events",
        do_stat=False,
        do_syst=True,
        symmetric_uncert=True,
        filename="TauPt_symmetric_uncert.png",
    )

    analysis.plot(
        val="TauPt",
        dataset="wtaunu",
        selection="SR_passID",
        xlabel=variable_data["TauPt"]["name"] + " [GeV]",
        ylabel="Events",
        do_stat=False,
        do_syst=True,
        symmetric_uncert=False,
        filename="TauPt_unsymmetric_uncert.png",
    )

    eff_sys = [Dataset.get_base_sys_name(s) for s in analysis["wtaunu"].eff_sys_set]
    sys_hists = [
        analysis["wtaunu"].histograms[NOMINAL_NAME]["SR_passID"][f"TauPt_{sys}_pct_uncert"]
        for sys in eff_sys
    ]
    cmap = plt.get_cmap("jet")
    colours = cmap(np.linspace(0, 1.0, len(eff_sys)))
    colours = [tuple(c) for c in colours]  # workaround mplhep bug
    analysis.plot(
        val=sys_hists,
        label=eff_sys,
        colour=colours,
        xlabel=variable_data["TauPt"]["name"] + " [GeV]",
        ylabel="pct uncertainty",
        kind="overlay",
        do_stat=False,
        do_syst=False,
        filename="TauPt_eff_uncerts.png",
    )

    tes_sys = [Dataset.get_base_sys_name(s) for s in analysis["wtaunu"].tes_sys_set]
    sys_hists = [
        analysis["wtaunu"].histograms[NOMINAL_NAME]["SR_passID"][f"TauPt_{sys}_pct_uncert"]
        for sys in tes_sys
    ]
    cmap = plt.get_cmap("jet")
    colours = cmap(np.linspace(0, 1.0, len(tes_sys)))
    colours = [tuple(c) for c in colours]  # workaround mplhep bug
    analysis.plot(
        val=sys_hists,
        label=tes_sys,
        colour=colours,
        xlabel=variable_data["TauPt"]["name"] + " [GeV]",
        ylabel="pct uncertainty",
        kind="overlay",
        do_stat=False,
        do_syst=False,
        filename="TauPt_tes_uncerts.png",
    )

    # desperate!
    all_sys = [
        Dataset.get_base_sys_name(s)
        for s in sorted(list(analysis["wtaunu"].eff_sys_set | analysis["wtaunu"].tes_sys_set))
    ]
    sys_map = {}
    for sys in all_sys:
        sys_map[sys] = ROOT_utils.get_th1_bin_values(
            analysis["wtaunu"].histograms[NOMINAL_NAME]["SR_passID"][f"TauPt_{sys}_tot_uncert"]
        )

    # analysis.plot(
    #     val=[hist_sys_down, hist_nom, hist_sys_up],
    #     xlabel=variable_data["TauPt"]["name"] + " [GeV]",
    #     label=["sys_down", "nominal", "sys_up"],
    #     ylabel="Events",
    #     kind="overlay",
    #     do_stat=False,
    #     do_sys=False,
    #     filename="TauPt_and_systematics.png",
    # )

    # # SYSTEMATIC UNCERTAINTIES
    # # ===========================================================================
    # sys_list = analysis["wtaunu"].sys_list  # list of systematic variations
    # default_args = {
    #     "label": sys_list,
    #     "do_stat": False,
    #     "do_sys": False,
    #     "ratio_plot": False,
    #     "legend_params": {"ncols": 1, "fontsize": 8},
    # }
    #
    # # for each sample
    # for selection in selections:
    #     for mc_sample in mc_samples:
    #         default_args["title"] = (
    #             f"{mc_sample} | "
    #             f"{selection} | "
    #             f"mc16d | "
    #             f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
    #         )
    #
    #         # mass variables
    #         for s in ("pct", "tot"):
    #             if s == "pct":
    #                 ylabel = "Percentage uncertainty / %"
    #             else:
    #                 ylabel = "Absolute uncertainty"
    #
    #             for v in measurement_vars_mass:
    #                 analysis.plot(
    #                     val=[
    #                         analysis[mc_sample].get_hist(
    #                             variable=f"{v}_{sys_name}_{s}_uncert",
    #                             systematic=NOMINAL_NAME,
    #                             selection=selection,
    #                         )
    #                         for sys_name in analysis["wtaunu"].sys_list
    #                     ],
    #                     logy=False,
    #                     logx=False,
    #                     ylabel=ylabel,
    #                     **default_args,
    #                     xlabel=variable_data[v]["name"] + " [GeV]",
    #                     filename=f"{v}_sys_{s}_uncert_liny.png",
    #                 )
    #
    #             # massless
    #             for v in measurement_vars_unitless:
    #                 analysis.plot(
    #                     val=[
    #                         analysis[mc_sample].get_hist(
    #                             variable=f"{v}_{sys_name}_{s}_uncert",
    #                             systematic=NOMINAL_NAME,
    #                             selection="SR_passID",
    #                         )
    #                         for sys_name in analysis["wtaunu"].sys_list
    #                     ],
    #                     logx=False,
    #                     logy=False,
    #                     ylabel=ylabel,
    #                     **default_args,
    #                     xlabel=variable_data[v]["name"],
    #                     filename=f"{v}_sys_{s}_uncert_liny.png",
    #                 )

    # analysis.histogram_printout()
    analysis.logger.info("DONE.")
