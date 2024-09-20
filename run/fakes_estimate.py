from pathlib import Path
from typing import Dict

import ROOT
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import Analysis
from src.cutting import Cut
from src.dataset import ProfileOpts
from src.histogram import Histogram1D, TH1_bin_edges
from utils.plotting_tools import get_axis_labels
from utils.variable_names import variable_data

DTA_PATH = Path("/data/DTA_outputs/2024-08-28/")
# DTA_PATH = Path("/data/DTA_outputs/2024-02-22/")
# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")
CUTFILE_DIR = Path("/afs/cern.ch/user/k/kghorban/framework/options/DTA_cuts/reco")

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        # SIGNAL
        # ====================================================================
        "wtaunu": {
            "data_path": {
                "lm_cut": DTA_PATH / "*Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
                "full": DTA_PATH / "*Sh_2211_Wtaunu_mW_120*/*.root",
            },
            "hard_cut": {"lm_cut": "TruthBosonM < 120"},
            "label": r"$W\rightarrow\tau\nu$",
            "is_signal": True,
        },
        # BACKGROUNDS
        # ====================================================================
        # W -> light lepton
        "wlv": {
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
        # Z -> ll
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
            "label": r"$Z\rightarrow (l/\nu)(l/\nu)$",
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
        # DIBOSON
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
        # DATA
        # ====================================================================
        "data": {
            # "data_path": DTA_PATH / "*data17*/*.root",
            "data_path": Path("/data/DTA_outputs/2024-03-05/*data17*/*.root"),
            "label": "data",
            "is_data": True,
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
        "SR_failID": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            fail_loose,
            pass_met150,
        ],
        "CR_passID": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            pass_loose,
            pass_100met,
        ],
        "CR_failID": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            fail_loose,
            pass_100met,
        ],
        # for MC
        "SR_passID_trueTau": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            pass_loose,
            pass_met150,
            pass_truetau,
        ],
        "SR_failID_trueTau": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            fail_loose,
            pass_met150,
            pass_truetau,
        ],
        "CR_passID_trueTau": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            pass_loose,
            pass_100met,
            pass_truetau,
        ],
        "CR_failID_trueTau": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            fail_loose,
            pass_100met,
            pass_truetau,
        ],
    }

    wanted_variables = {
        "TauEta",
        "TauPhi",
        "TauPt",
        "MET_met",
        "MET_phi",
        "MTW",
        "DeltaPhi_tau_met",
        "TauPt_div_MET",
        "TauRNNJetScore",
        "TauBDTEleScore",
        "TruthTauPt",
        "TruthTauEta",
        "TruthTauPhi",
        "TauNCoreTracks",
        # "TruthTau_nChargedTracks",
        # "TruthTau_nNeutralTracks",
        "TauPt_res",
        "TauPt_diff",
        "MatchedTruthParticlePt",
        "MatchedTruthParticle_isTau",
        "MatchedTruthParticle_isElectron",
        "MatchedTruthParticle_isMuon",
        "MatchedTruthParticle_isPhoton",
        "MatchedTruthParticle_isJet",
        "nJets",
    }
    measurement_vars_mass = [
        "TauPt",
        "MTW",
        "MET_met",
    ]
    measurement_vars_unitless = [
        "TauEta",
        "TauPhi",
        "nJets",
        "TauNCoreTracks",
        "TauRNNJetScore",
        "TauBDTEleScore",
    ]
    measurement_vars = measurement_vars_unitless + measurement_vars_mass
    profile_vars = [
        "TauPt_res",
        "TauPt_diff",
        "MatchedTruthParticlePt",
        "MatchedTruthParticle_isTau",
        "MatchedTruthParticle_isElectron",
        "MatchedTruthParticle_isMuon",
        "MatchedTruthParticle_isPhoton",
        "MatchedTruthParticle_isJet",
    ]
    # define which profiles to calculate
    profiles: dict[str, ProfileOpts] = dict()
    for measurement_var in measurement_vars:
        for prof_var in profile_vars:
            profiles[f"{measurement_var}_{prof_var}"] = ProfileOpts(
                x=measurement_var,
                y=prof_var,
                weight="" if "MatchedTruthParticle" in prof_var else "reco_weight",
            )

    # RUN
    # ========================================================================
    analysis = Analysis(
        datasets,
        year=2017,
        regen_histograms=True,
        # do_systematics=True,
        # regen_metadata=True,
        ttree="T_s1thv_NOMINAL",
        cuts=selections,
        analysis_label="fakes_estimate",
        dataset_type="dta",
        log_level=10,
        log_out="both",
        extract_vars=wanted_variables,
        import_missing_columns_as_nan=True,
        profiles=profiles,
        binnings={
            "": {
                "MTW": np.geomspace(150, 1000, 21),
                "TauPt": np.geomspace(170, 1000, 21),
                "TauEta": np.linspace(-2.5, 2.5, 21),
                "EleEta": np.linspace(-2.5, 2.5, 21),
                "MuonEta": np.linspace(-2.5, 2.5, 21),
                "MET_met": np.geomspace(150, 1000, 21),
                "DeltaPhi_tau_met": np.linspace(0, 3.5, 21),
                "TauPt_div_MET": np.linspace(0, 3, 61),
                "TauRNNJetScore": np.linspace(0, 1, 51),
                "TauBDTEleScore": np.linspace(0, 1, 51),
                "TruthTauPt": np.geomspace(1, 1000, 21),
                "TauNCoreTracks": np.linspace(0, 4, 5),
                "TauPt_res": np.linspace(-1, 1, 51),
                "TauPt_diff": np.linspace(-300, 300, 51),
                "badJet": (2, 0, 2),
            },
            "CR_failID": {
                "MET_met": np.geomspace(1, 100, 51),
            },
            "CR_passID": {
                "MET_met": np.geomspace(1, 100, 51),
            },
            "CR_failID_trueTau": {
                "MET_met": np.geomspace(1, 100, 51),
            },
            "CR_passID_trueTau": {
                "MET_met": np.geomspace(1, 100, 51),
            },
        },
    )
    all_samples = [analysis.data_sample] + analysis.mc_samples
    mc_samples = analysis.mc_samples
    analysis.full_cutflow_printout(datasets=all_samples)
    analysis.print_metadata_table(datasets=mc_samples)
    for dataset in analysis:
        ROOT.RDF.SaveGraph(
            dataset.filters["SR_passID"].df,
            f"{analysis.paths.output_dir}/{dataset.name}_SR_graph.dot",
        )

    # set colours for samples
    c_iter = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for ds in mc_samples:
        c = next(c_iter)
        analysis[ds].colour = c
    analysis["data"].colour = "k"
    fakes_colour = next(c_iter)

    # FAKES ESTIMATE
    # ========================================================================
    for fakes_filter, fakes_filter_name in [
        ("TauNCoreTracks == 1", "1prong"),
        ("TauNCoreTracks == 3", "3prong"),
        ("", ""),
    ]:
        # define new selection for 1- or 3- tau prongs
        if fakes_filter:
            for selection in [
                "CR_passID",
                "CR_failID",
                "SR_passID",
                "SR_failID",
                "CR_passID_trueTau",
                "CR_failID_trueTau",
                "SR_passID_trueTau",
                "SR_failID_trueTau",
            ]:
                for ds in analysis.datasets:
                    analysis[ds].define_selection(
                        fakes_filter, fakes_filter_name + "_" + selection, selection
                    )
        if fakes_filter_name:
            prefix = fakes_filter_name + "_"
        else:
            prefix = ""

        for fakes_source in ["TauPt", "MTW"]:
            analysis.do_fakes_estimate(
                fakes_source,
                measurement_vars,
                f"{prefix}CR_passID",
                f"{prefix}CR_failID",
                f"{prefix}SR_passID",
                f"{prefix}SR_failID",
                f"{prefix}CR_passID_trueTau",
                f"{prefix}CR_failID_trueTau",
                f"{prefix}SR_passID_trueTau",
                f"{prefix}SR_failID_trueTau",
                name=fakes_filter_name,
                save_intermediates=True,
            )

            # Intermediates
            # ----------------------------------------------------------------------------
            CR_passID_data = analysis.get_hist(fakes_source, "data", f"{prefix}CR_passID")
            CR_failID_data = analysis.get_hist(fakes_source, "data", f"{prefix}CR_failID")
            SR_failID_data = analysis.get_hist(fakes_source, "data", f"{prefix}SR_failID")
            CR_passID_mc = analysis.get_hist(
                f"{prefix}all_mc_{fakes_source}_{prefix}CR_passID_trueTau"
            )
            CR_failID_mc = analysis.get_hist(
                f"{prefix}all_mc_{fakes_source}_{prefix}CR_failID_trueTau"
            )
            SR_failID_mc = analysis.get_hist(
                f"{prefix}all_mc_{fakes_source}_{prefix}SR_failID_trueTau"
            )

            analysis.plot(
                [CR_passID_data, CR_failID_data, CR_passID_mc, CR_failID_mc],
                label=["CR_passID_data", "CR_failID_data", "CR_passID_mc", "CR_failID_mc"],
                yerr=False,
                logy=True,
                xlabel=fakes_source,
                ratio_plot=False,
                filename=f"{prefix}FF_histograms_{fakes_source}.png",
            )
            analysis.plot(
                [CR_failID_data - CR_failID_mc, CR_passID_data - CR_passID_mc],
                label=["CR_failID_data - CR_failID_mc", "CR_passID_data - CR_passID_mc"],
                yerr=False,
                logy=True,
                xlabel=fakes_source,
                ratio_plot=True,
                filename=f"{prefix}FF_histograms_diff_{fakes_source}.png",
                ratio_label="Fake Factor",
            )
            analysis.plot(
                [SR_failID_data, SR_failID_mc],
                label=["SR_failID_data", "SR_failID_mc"],
                yerr=False,
                logy=True,
                xlabel=fakes_source,
                ratio_plot=False,
                filename=f"{prefix}FF_calculation_{fakes_source}.png",
            )
            analysis.plot(
                SR_failID_data - SR_failID_mc,
                label=["SR_failID_data - SR_failID_mc"],
                yerr=False,
                logy=True,
                xlabel=fakes_source,
                ratio_plot=False,
                filename=f"{prefix}FF_calculation_delta_SR_fail_{fakes_source}.png",
            )

            # Fake factors
            # ----------------------------------------------------------------------------
            analysis.plot(
                val=f"{prefix}{fakes_source}_FF",
                xlabel=r"$p_T^\tau$ [GeV]" if fakes_source == "TauPt" else r"$m_T^W$ [GeV]",
                yerr=False,
                selection=None,
                logx=False,
                logy=False,
                ylabel="Fake factor",
                filename=f"{prefix}{fakes_source}_FF.png",
            )

            # Stacks with Fakes background
            # ----------------------------------------------------------------------------
            # log axes
            default_args = {
                "dataset": all_samples + [None],
                "label": [analysis[ds].label for ds in all_samples] + ["Multijet"],
                "colour": [analysis[ds].colour for ds in all_samples] + [fakes_colour],
                "title": f"{fakes_source} fakes binning | data17 | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                "yerr": True,
                "suffix": "fake_scaled_log",
                "ratio_plot": True,
                "ratio_axlim": (0.8, 1.2),
                "kind": "stack",
            }

            def FF_vars(s: str) -> list[str]:
                """List of variable names for each sample"""
                return (
                    [f"data_{s}_{prefix}SR_passID"]
                    + [f"{ds_}_{s}_{prefix}SR_passID_trueTau" for ds_ in mc_samples]
                    + [f"{prefix}{s}_fakes_bkg_{fakes_source}_src"]
                )

            # make sure the histogram exists
            if fakes_filter:
                for v in measurement_vars:
                    analysis.gen_histogram(v, analysis.data_sample, prefix + "SR_passID")
                    for d in analysis.mc_samples:
                        analysis.gen_histogram(v, d, prefix + "SR_passID_trueTau")

            # mass variables
            for v in measurement_vars_mass:
                analysis.plot(
                    val=FF_vars(v),
                    logx=True,
                    logy=True,
                    **default_args,
                    xlabel=variable_data[v]["name"] + " [GeV]",
                    filename=f"{prefix}{v}_fakes_stack_{fakes_source}_log.png",
                )
                analysis.plot(
                    val=FF_vars(v),
                    logx=False,
                    logy=False,
                    **default_args,
                    xlabel=variable_data[v]["name"] + " [GeV]",
                    filename=f"{prefix}{v}_fakes_stack_{fakes_source}_liny.png",
                )
            # massless
            for v in measurement_vars_unitless:
                analysis.plot(
                    val=FF_vars(v),
                    **default_args,
                    logy=True,
                    logx=False,
                    xlabel=variable_data[v]["name"],
                    filename=f"{prefix}{v}_fakes_stack_{fakes_source}_log.png",
                )
                analysis.plot(
                    val=FF_vars(v),
                    **default_args,
                    logy=False,
                    logx=False,
                    xlabel=variable_data[v]["name"],
                    filename=f"{prefix}{v}_fakes_stack_{fakes_source}_liny.png",
                )

    # compare fake factors
    for fakes_source in ["MTW", "TauPt"]:
        analysis.plot(
            val=[f"1prong_{fakes_source}_FF", f"3prong_{fakes_source}_FF"],
            label=["1-prong", "3-prong"],
            xlabel=r"$p_T^\tau$ [GeV]" if fakes_source == "TauPt" else r"$m_T^W$ [GeV]",
            yerr=False,
            selection=None,
            logx=False,
            logy=False,
            ylabel="Fake factor",
            filename=f"{fakes_source}_FF_compare.png",
        )

    # Direct data scaling comparison
    # ----------------------------------------------------------------------------
    # log axes
    default_args = {
        "title": f"data17 | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
        "label": ["Data SR", "MC + TauPt Fakes", "MC + MTW Fakes"],
        "colour": ["k", "b", "r"],
        "yerr": True,
        "suffix": "fake_scaled_log",
        "ratio_plot": True,
        "ratio_axlim": (0.8, 1.2),
    }

    def FF_full_bkg(s: str, t: str) -> Histogram1D:
        """Sum of all backgrounds + signal + FF"""
        return Histogram1D(
            th1=analysis.sum_hists(
                [analysis.get_hist_name(s, ds_, "SR_passID_trueTau") for ds_ in mc_samples]
            )
            + analysis.get_hist(f"{s}_fakes_bkg_{t}_src")
        )

    # mass variables
    for v in measurement_vars_mass:
        analysis.plot(
            val=[
                f"data_{v}_SR_passID",
                FF_full_bkg(v, "TauPt"),
                FF_full_bkg(v, "MTW"),
            ],
            logy=True,
            logx=True,
            **default_args,
            xlabel=variable_data[v]["name"] + " [GeV]",
            filename=f"FF_compare_{v}_log.png",
        )
        analysis.plot(
            val=[
                f"data_{v}_SR_passID",
                FF_full_bkg(v, "TauPt"),
                FF_full_bkg(v, "MTW"),
            ],
            logy=False,
            logx=False,
            **default_args,
            xlabel=variable_data[v]["name"] + " [GeV]",
            filename=f"FF_compare_{v}_liny.png",
        )
    # massless
    for v in measurement_vars_unitless:
        analysis.plot(
            val=[
                f"data_{v}_SR_passID",
                FF_full_bkg(v, "TauPt"),
                FF_full_bkg(v, "MTW"),
            ],
            logy=True,
            logx=False,
            **default_args,
            xlabel=variable_data[v]["name"],
            filename=f"FF_compare_{v}_log.png",
        )
        analysis.plot(
            val=[
                f"data_{v}_SR_passID",
                FF_full_bkg(v, "TauPt"),
                FF_full_bkg(v, "MTW"),
            ],
            logx=False,
            logy=False,
            **default_args,
            xlabel=variable_data[v]["name"],
            filename=f"FF_compare_{v}_liny.png",
        )

    # No Fakes
    # ----------------------------------------------------------------------------
    default_args = {
        "dataset": all_samples,
        "title": f"data17 | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
        "yerr": True,
        "selection": "SR_passID",
        "ratio_plot": True,
        "ratio_axlim": (0.8, 1.2),
        "kind": "stack",
    }

    # mass-like variables
    for var in measurement_vars_mass:
        analysis.plot(
            val=var,
            **default_args,
            logx=True,
            logy=True,
            filename=f"{var}_stack_no_fakes_log.png",
        )
        analysis.plot(
            val=var,
            **default_args,
            logy=False,
            logx=False,
            filename=f"{var}_stack_no_fakes_liny.png",
        )

    # unitless variables
    for var in measurement_vars_unitless:
        analysis.plot(
            val=var,
            **default_args,
            logx=False,
            logy=True,
            filename=f"{var}_stack_no_fakes_log.png",
        )
        analysis.plot(
            val=var,
            **default_args,
            logy=False,
            logx=False,
            filename=f"{var}_stack_no_fakes_liny.png",
        )

    # Fakes distribution across kinematic variable for signal MC
    # -----------------------------------------------------------------------
    for var in measurement_vars:
        xlabel = get_axis_labels(var)[0]
        sel = "SR_passID"
        mc = "wtaunu"

        # for all MC
        wtaunu_el_fakes = analysis.sum_hists(
            [f"{s}_{var}_MatchedTruthParticle_isElectron_{sel}_PROFILE" for s in mc_samples],
            f"all_mc_{var}_MatchedTruthParticle_isElectron_{sel}_PROFILE",
        )
        wtaunu_mu_fakes = analysis.sum_hists(
            [f"{s}_{var}_MatchedTruthParticle_isMuon_{sel}_PROFILE" for s in mc_samples],
            f"all_mc_{var}_MatchedTruthParticle_isMuon_{sel}_PROFILE",
        )
        wtaunu_ph_fakes = analysis.sum_hists(
            [f"{s}_{var}_MatchedTruthParticle_isPhoton_{sel}_PROFILE" for s in mc_samples],
            f"all_mc_{var}_MatchedTruthParticle_isPhoton_{sel}_PROFILE",
        )
        wtaunu_jet_fakes = analysis.sum_hists(
            [f"{s}_{var}_MatchedTruthParticle_isJet_{sel}_PROFILE" for s in mc_samples],
            f"all_mc_{var}_MatchedTruthParticle_isJet_{sel}_PROFILE",
        )
        wtaunu_true_taus = analysis.sum_hists(
            [f"{s}_{var}_MatchedTruthParticle_isTau_{sel}_PROFILE" for s in mc_samples],
            f"all_mc_{var}_MatchedTruthParticle_isTau_{sel}_PROFILE",
        )
        analysis.plot(
            [wtaunu_jet_fakes, wtaunu_ph_fakes, wtaunu_mu_fakes, wtaunu_el_fakes, wtaunu_true_taus],
            label=[
                "Jet-matched fake taus",
                "Photon-matched fake taus",
                "Muon-matched fake taus",
                "electron-matched fake taus",
                "True taus",
            ],
            sort=False,
            yerr=False,
            colour=list(plt.rcParams["axes.prop_cycle"].by_key()["color"])[:5],
            title=f"Fake fractions for {var} in {sel} for all MC in SR",
            y_axlim=(0, 1),
            kind="stack",
            xlabel=xlabel,
            ylabel="Fraction of fake matched taus in signal MC",
            filename=f"all_mc_{var}_{sel}_fake_fractions.png",
        )

        # for all MC
        el_fakes = analysis.histograms[f"{mc}_{var}_MatchedTruthParticle_isElectron_{sel}_PROFILE"]
        mu_fakes = analysis.histograms[f"{mc}_{var}_MatchedTruthParticle_isMuon_{sel}_PROFILE"]
        ph_fakes = analysis.histograms[f"{mc}_{var}_MatchedTruthParticle_isPhoton_{sel}_PROFILE"]
        jet_fakes = analysis.histograms[f"{mc}_{var}_MatchedTruthParticle_isJet_{sel}_PROFILE"]
        true_taus = analysis.histograms[f"{mc}_{var}_MatchedTruthParticle_isTau_{sel}_PROFILE"]

        sel_hist = analysis.get_hist(var, "wtaunu", sel, TH1=True)
        nbins = sel_hist.GetNbinsX()
        bin_edges = TH1_bin_edges(sel_hist)

        analysis.plot(
            [jet_fakes, ph_fakes, mu_fakes, el_fakes, true_taus],
            label=[
                "Jet Fakes",
                "Photon Fakes",
                "Muon Fakes",
                "Electron Fakes",
                "True taus",
            ],
            sort=False,
            yerr=False,
            colour=list(plt.rcParams["axes.prop_cycle"].by_key()["color"])[:5],
            title=f"Fake fractions for {var} in {mc} for SR",
            y_axlim=(0, 1),
            kind="stack",
            xlabel=xlabel,
            ylabel="Fraction of fake matched taus in signal MC",
            filename=f"{mc}_{var}_{sel}_fake_fractions.png",
        )

    # truth hists for mental health
    # ----------------------------------------------------------------------------
    default_args = {
        "dataset": mc_samples,
        "title": f"Truth Taus | mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
        "yerr": True,
        "selection": False,
        "ratio_plot": False,
        "stats_box": False,
    }
    analysis.plot(val="MatchedTruthParticlePt", **default_args, logx=True)
    analysis.plot(val="TruthTauPt", **default_args, logx=True)
    analysis.plot(val="TruthTauEta", **default_args)
    analysis.plot(val="TruthTauPhi", **default_args)

    default_args["selection"] = "SR_passID_trueTau"
    analysis.plot(val="MatchedTruthParticlePt", **default_args, logx=True)
    analysis.plot(val="TruthTauPt", **default_args, logx=True)
    analysis.plot(val="TruthTauEta", **default_args)
    analysis.plot(val="TruthTauPhi", **default_args)

    # tau pt resolution
    analysis.plot(
        val="TauPt_res",
        dataset="wtaunu",
        xlabel=r"$(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
        title=(
            r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
            + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
        ),
        filename="wtaunu_taupt_resolution.png",
    )
    analysis.plot(
        val="TauPt_diff",
        dataset="wtaunu",
        xlabel=r"$p_T^\mathrm{true} - p_T^\mathrm{reco}$ [GeV]",
        title=(
            r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
            + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
        ),
        filename="wtaunu_taupt_truthrecodiff.png",
    )
    analysis.plot(
        val="MTW_TauPt_res_SR_passID_PROFILE",
        dataset="wtaunu",
        ylabel=r"Mean $(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
        xlabel=r"$m_W^T$ [GeV]",
        title=(
            r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
            + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
        ),
        y_axlim=(-10, 10),
        filename="wtaunu_mtw_taupt_profile.png",
    )
    analysis.plot(
        val=[
            "TauPt_res_SR_passID_trueTau",
            "TauPt_res_SR_failID_trueTau",
            "TauPt_res_CR_passID_trueTau",
            "TauPt_res_CR_failID_trueTau",
        ],
        dataset="wtaunu",
        label=["SR_passID", "SR_failID", "CR_passID", "CR_failID"],
        xlabel=r"$(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
        title=(
            r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
            + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
        ),
        ratio_plot=False,
        filename="wtaunu_taupt_resolution_selections.png",
    )
    analysis.plot(
        val=[
            "TauPt_diff_SR_passID_trueTau",
            "TauPt_diff_SR_failID_trueTau",
            "TauPt_diff_CR_passID_trueTau",
            "TauPt_diff_CR_failID_trueTau",
        ],
        dataset="wtaunu",
        label=["SR_passID", "SR_failID", "CR_passID", "CR_failID"],
        xlabel=r"$p_T^\mathrm{true} - p_T^\mathrm{reco}$ [GeV]",
        title=(
            r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
            + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
        ),
        ratio_plot=False,
        filename="wtaunu_taupt_truthrecodiff_selections.png",
    )
    analysis.plot(
        val=[
            "MTW_TauPt_res_SR_passID_trueTau_PROFILE",
            "MTW_TauPt_res_SR_failID_trueTau_PROFILE",
            "MTW_TauPt_res_CR_passID_trueTau_PROFILE",
            "MTW_TauPt_res_CR_failID_trueTau_PROFILE",
        ],
        dataset="wtaunu",
        label=["SR_passID", "SR_failID", "CR_passID", "CR_failID"],
        ylabel=r"Mean $(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
        xlabel=r"$m_W^T$ [GeV]",
        title=(
            r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
            + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
        ),
        y_axlim=(-10, 10),
        ratio_plot=False,
        filename="wtaunu_mtw_taupt_profile_selections.png",
    )

    # analysis.histogram_printout()
    analysis.save_histograms()
    analysis.logger.info("DONE.")
