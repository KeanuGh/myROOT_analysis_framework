from pathlib import Path

import numpy as np
from binnings import BINNINGS, nedges
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples

from src.analysis import Analysis
from src.cutting import Cut
from utils.helper_functions import smart_join
from utils.plotting_tools import PlotKwargs
from utils.variable_names import variable_data

YEAR = 2017
LOAD_SAVED_HISTS = True

# CUTS & SELECTIONS
# ========================================================================
pass_presel = Cut(
    r"Pass preselection",
    r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1) && passMetTrigger && (badJet == 0)"
    r"&& ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)"
    r"&& ((TauNCoreTracks == 1) || (TauNCoreTracks == 3))",
)
pass_taupt170 = Cut(
    r"$p_T^\tau > 170$",
    r"TauPt > 170",
)
pass_eta = Cut(
    r"$|\eta^{\tau_\mathrm{had-vis}}| < 1.37 || 1.52 < |\eta^{\tau_\mathrm{had-vis}}| < 2.47$",
    r"(((abs(TauEta) < 1.37) || (1.52 < abs(TauEta))) && (abs(TauEta) < 2.47))",
)
pass_mtw350 = Cut(
    r"$m_T^W > 350$",
    r"MTW > 350",
)
pass_loose = Cut(
    r"\mathrm{Pass Loose ID}",
    r"(TauBDTEleScore > 0.05) && "
    r"((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
)
fail_loose = Cut(
    r"\mathrm{Fail Loose ID}",
    r"(TauBDTEleScore > 0.05) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
)
pass_medium = Cut(
    r"\mathrm{Pass Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
fail_medium = Cut(
    r"\mathrm{Fail Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
pass_tight = Cut(
    r"\mathrm{Pass Tight ID}",
    r"(TauBDTEleScore > 0.15) && "
    r"((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
)
fail_tight = Cut(
    r"\mathrm{Fail Tight ID}",
    r"(TauBDTEleScore > 0.15) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
)
pass_met170 = Cut(
    r"$E_T^{\mathrm{miss}} > 170$",
    r"MET_met > 170",
)
pass_150met = Cut(
    r"$E_T^{\mathrm{miss}} < 150$",
    r"MET_met < 150",
)
pass_1prong = Cut(
    "1-prong",
    "TauNCoreTracks == 1",
)
pass_3prong = Cut(
    "3-prong",
    "TauNCoreTracks == 3",
)
pass_truetau = Cut(
    # this is a multijet background estimation. Tau is "True" in this case if it is a lepton
    r"True Tau",
    "MatchedTruthParticle_isHadronicTau == true || MatchedTruthParticle_isMuon == true || MatchedTruthParticle_isElectron == true",
)
fail_truetau = Cut(
    r"Fake Tau",
    "!(MatchedTruthParticle_isHadronicTau == true || MatchedTruthParticle_isMuon == true || MatchedTruthParticle_isElectron == true)",
)

# selections
selections_loose: dict[str, list[Cut]] = {
    "loose_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_loose,
        pass_met170,
    ],
    "loose_SR_failID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        fail_loose,
        pass_met170,
    ],
    "loose_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_loose,
        pass_150met,
    ],
    "loose_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        fail_loose,
        pass_150met,
    ],
}
selections_medium: dict[str, list[Cut]] = {
    "medium_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_medium,
        pass_met170,
    ],
    "medium_SR_failID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        fail_medium,
        pass_met170,
    ],
    "medium_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_medium,
        pass_150met,
    ],
    "medium_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        fail_medium,
        pass_150met,
    ],
}
selections_tight: dict[str, list[Cut]] = {
    "tight_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_tight,
        pass_met170,
    ],
    "tight_SR_failID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        fail_tight,
        pass_met170,
    ],
    "tight_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_tight,
        pass_150met,
    ],
    "tight_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        fail_tight,
        pass_150met,
    ],
}
selections = selections_loose | selections_medium | selections_tight

# define selection for MC samples
selections_list = list(selections.keys())
selections_cuts = list(selections.values())
for selection, cut_list in zip(selections_list, selections_cuts, strict=True):
    selections[f"trueTau_{selection}"] = cut_list + [pass_truetau]
    # define selections for 1- or 3- tau prongs
    for cutstr, cut_name in [
        ("TauNCoreTracks == 1", "1prong"),
        ("TauNCoreTracks == 3", "3prong"),
        ("TauCharge == 1", "tauplus"),
        ("TauCharge == -1", "tauminus"),
    ]:
        selections[f"{cut_name}_{selection}"] = cut_list + [Cut(cut_name, cutstr)]
        selections[f"trueTau_{cut_name}_{selection}"] = cut_list + [
            pass_truetau,
            Cut(cut_name, cutstr),
        ]
# for data
selections_notruth = {n: s for n, s in selections.items() if not n.startswith("trueTau_")}

# VARIABLES
# ========================================================================
wanted_variables = {
    "TauEta",
    "TauPhi",
    "TauPt",
    "MET_met",
    "MET_phi",
    "MTW",
    "TauRNNJetScore",
    "TauBDTEleScore",
    "TauNCoreTracks",
    "nJets",
    "AbsDeltaPhi_tau_met",
    "TauPt_div_MET",
}
measurement_vars_mass = [
    "TauPt",
    "MTW",
    "MET_met",
]
measurement_vars_unitless = [
    "TauEta",
    "TauPhi",
    "TauBDTEleScore",
    "TauRNNJetScore",
    # "DeltaPhi_tau_met",
    "AbsDeltaPhi_tau_met",
    "TauPt_div_MET",
    "TauNCoreTracks",
]
measurement_vars = measurement_vars_unitless + measurement_vars_mass
datasets = analysis_samples(selections, data_selections=selections_notruth, snapshot=True)


def run_analysis() -> Analysis:
    """Run analysis"""
    return Analysis(
        datasets,
        year=YEAR,
        rerun=not LOAD_SAVED_HISTS,
        regen_histograms=not LOAD_SAVED_HISTS,
        do_systematics=False,
        # regen_metadata=True,
        metadata_cache=DSID_METADATA_CACHE,
        # output_dir="/eos/home-k/kghorban/framework_outputs/analysis_main",
        ttree=NOMINAL_NAME,
        analysis_label=Path(__file__).stem,
        log_level=10,
        log_out="both",
        extract_vars=wanted_variables,
        import_missing_columns_as_nan=True,
        histogram_vars=set(measurement_vars),
        systematics_for_selection={".*SR_passID"},
        skip_sys={
            r".*TAUS_TRUEHADTAU_EFF_RNNID_.*",
            r".*TAUS_TRUEHADTAU_EFF_JETID_.*",
        },
        binnings={
            "": BINNINGS,
            ".*_CR_.*ID": {
                "MET_met": np.geomspace(1, 100, nedges),
            },
        },
    )


if __name__ == "__main__":
    # RUN
    # ========================================================================
    analysis = run_analysis()
    if LOAD_SAVED_HISTS:
        analysis.load_hists()

    base_plotting_dir = analysis.paths.plot_dir
    all_samples = [analysis.data_sample] + analysis.mc_samples
    mc_samples = analysis.mc_samples
    if not LOAD_SAVED_HISTS:
        analysis.full_cutflow_printout(datasets=all_samples)
        analysis.print_metadata_table(datasets=mc_samples)
        for mc in mc_samples:
            analysis[mc].calculate_systematic_uncertainties()
    fakes_colour = next(analysis.c_iter)

    wps = (
        "loose",
        "medium",
        "tight",
    )
    sections = (
        "tauminus",
        "tauplus",
        "1prong",
        "3prong",
        "",
    )
    fakes_sources = (
        "MTW",
        "TauPt",
    )
    source_colours = [
        "darkviolet",
        "mediumblue",
    ]

    # START OF WP LOOP
    # ========================================================================
    # loop over each working point
    for wp in wps:
        wp_dir = base_plotting_dir / wp

        # FAKES ESTIMATE
        # ========================================================================
        for sec_str in sections:
            sec = sec_str + "_" if sec_str else ""

            for fakes_source in fakes_sources:
                if not LOAD_SAVED_HISTS:
                    analysis.do_fakes_estimate(
                        fakes_source,
                        measurement_vars,
                        f"{sec}{wp}_CR_passID",
                        f"{sec}{wp}_CR_failID",
                        f"{sec}{wp}_SR_passID",
                        f"{sec}{wp}_SR_failID",
                        f"trueTau_{sec}{wp}_CR_passID",
                        f"trueTau_{sec}{wp}_CR_failID",
                        f"trueTau_{sec}{wp}_SR_passID",
                        f"trueTau_{sec}{wp}_SR_failID",
                        name=f"{sec}{wp}",
                        systematic=NOMINAL_NAME,
                        save_intermediates=True,
                    )

                # Intermediates
                # ----------------------------------------------------------------------------
                CR_passID_data = analysis.get_hist(
                    fakes_source,
                    "data",
                    systematic=NOMINAL_NAME,
                    selection=f"{sec}{wp}_CR_passID",
                )
                CR_failID_data = analysis.get_hist(
                    fakes_source,
                    "data",
                    systematic=NOMINAL_NAME,
                    selection=f"{sec}{wp}_CR_failID",
                )
                SR_failID_data = analysis.get_hist(
                    fakes_source,
                    "data",
                    systematic=NOMINAL_NAME,
                    selection=f"{sec}{wp}_SR_failID",
                )
                CR_passID_mc = analysis.get_hist(
                    f"{sec}{wp}_all_mc_{fakes_source}_trueTau_{sec}{wp}_CR_passID"
                )
                CR_failID_mc = analysis.get_hist(
                    f"{sec}{wp}_all_mc_{fakes_source}_trueTau_{sec}{wp}_CR_failID"
                )
                SR_failID_mc = analysis.get_hist(
                    f"{sec}{wp}_all_mc_{fakes_source}_trueTau_{sec}{wp}_SR_failID"
                )
                default_args: PlotKwargs = {
                    "do_stat": False,
                    "logx": True,
                    "logy": True,
                    "xlabel": variable_data[fakes_source]["name"] + " [GeV]",
                    "ylabel": "Weighted events",
                    "title": smart_join(
                        f"{variable_data[fakes_source]['name']} fakes binning",
                        wp.title(),
                        str(YEAR),
                        f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                        sep=" | ",
                    ),
                }

                analysis.paths.plot_dir = wp_dir / "fakes_intermediates"
                analysis.plot(
                    [CR_passID_data, CR_failID_data, CR_passID_mc, CR_failID_mc],
                    label=[
                        r"$N^{\mathrm{CR}}_{\mathrm{passID,data}}$",
                        r"$N^{\mathrm{CR}}_{\mathrm{failID,data}}$",
                        r"$N^{\mathrm{CR}}_{\mathrm{passID,MC}}$",
                        r"$N^{\mathrm{CR}}_{\mathrm{failID,MC}}$",
                    ],
                    **default_args,
                    filename=f"{sec}{wp}_FF_histograms_{fakes_source}.png",
                )
                analysis.plot(
                    [CR_failID_data - CR_failID_mc, CR_passID_data - CR_passID_mc],
                    label=[
                        r"$N^{\mathrm{CR}}_{\mathrm{failID,data}} - N^{\mathrm{CR}}_{\mathrm{failID,MC}}$",
                        r"$N^{\mathrm{CR}}_{\mathrm{passID,data}} - N^{\mathrm{CR}}_{\mathrm{passID,MC}}$",
                    ],
                    **default_args,
                    ratio_plot=True,
                    filename=f"{sec}{wp}_FF_histograms_diff_{fakes_source}.png",
                    ratio_label="FF",
                )
                analysis.plot(
                    [SR_failID_data, SR_failID_mc],
                    label=["SR_failID_data", "SR_failID_mc"],
                    **default_args,
                    filename=f"{sec}{wp}_FF_calculation_{fakes_source}.png",
                )
                analysis.plot(
                    SR_failID_data - SR_failID_mc,
                    label="SR_failID_data - SR_failID_mc",
                    **default_args,
                    filename=f"{sec}{wp}_FF_calculation_delta_SR_fail_{fakes_source}.png",
                )

                # Fake factors
                # ----------------------------------------------------------------------------
                analysis.paths.plot_dir = wp_dir / "fake_factors"
                analysis.plot(
                    val=f"{sec}{wp}_{fakes_source}_FF",
                    xlabel=r"$p_T^\tau$ [GeV]" if fakes_source == "TauPt" else r"$m_T^W$ [GeV]",
                    do_stat=False,
                    logx=False,
                    logy=False,
                    ylabel="Fake factor",
                    filename=f"{sec}{wp}_{fakes_source}_FF.png",
                )

                # Stacks with Fakes background
                # ----------------------------------------------------------------------------
                analysis.paths.plot_dir = wp_dir / "fakes_stacks"
                # log axes
                default_args: PlotKwargs = {
                    "dataset": all_samples + [None],
                    "systematic": NOMINAL_NAME,
                    "selection": (
                            [f"{sec}{wp}_SR_passID"]
                            + [f"{sec}{wp}_SR_passID"] * len(mc_samples)
                            + [None]
                    ),
                    "label": [analysis[ds].label for ds in all_samples] + ["Fake Jet Estimate"],
                    "colour": [analysis[ds].colour for ds in all_samples] + [fakes_colour],
                    "title": smart_join(
                        f"{wp.title()} ID SR",
                        str(YEAR),
                        f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                        sep=" | ",
                    ),
                    "ylabel": "Events",
                    "do_stat": True,
                    "do_syst": False,
                    "ratio_plot": True,
                    "ratio_axlim": (0.5, 1.5),
                    "kind": "stack",
                }


                # mass variables
                for v in measurement_vars:
                    if v in measurement_vars_mass:
                        default_args.update(
                            {"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"}
                        )
                    elif v in measurement_vars_unitless:
                        default_args.update({"logx": False, "xlabel": variable_data[v]["name"]})
                    ff_vals = [v] * len(all_samples) + [
                        f"{sec}{wp}_{v}_fakes_bkg_{fakes_source}_src"
                    ]
                    analysis.plot(
                        val=ff_vals,
                        **default_args,
                        logy=True,
                        filename=f"{sec}{wp}_{v}_fakes_stack_{fakes_source}_log.png",
                    )
                    analysis.plot(
                        val=ff_vals,
                        **default_args,
                        logy=False,
                        filename=f"{sec}{wp}_{v}_fakes_stack_{fakes_source}_liny.png",
                    )

        # Fake factors
        # ----------------------------------------------------------------------------
        analysis.paths.plot_dir = wp_dir / "fakes_comparisons"
        for fakes_source in fakes_sources:
            analysis.plot(
                val=[
                    f"1prong_{wp}_{fakes_source}_FF",
                    f"3prong_{wp}_{fakes_source}_FF",
                    f"{wp}_{fakes_source}_FF",
                ],
                label=[
                    "1-prong",
                    "3-prong",
                    "1 + 3 prong",
                ],
                title=smart_join(
                    f"{wp.title()} ID",
                    str(YEAR),
                    f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                    sep=" | ",
                ),
                xlabel=r"$p_T^\tau$ [GeV]" if fakes_source == "TauPt" else r"$m_T^W$ [GeV]",
                do_stat=True,
                logx=True,
                logy=False,
                ylabel="Fake factor",
                filename=f"{wp}_{fakes_source}_FF_prong_compare.png",
            )

    if not LOAD_SAVED_HISTS:
        analysis.save_hists()
    analysis.histogram_printout(to_file="txt")
    analysis.logger.info("DONE.")
