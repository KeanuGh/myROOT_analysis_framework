from typing import List, Tuple

import ROOT
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from histogram import Histogram1D
from src.cutfile import Cutfile
from utils import ROOT_utils
from utils.plotting_tools import set_axis_options, get_TH1_bins
from utils.var_helpers import derived_vars
from utils.variable_names import variable_data


def plot_rdf(
    rdf: ROOT.RDataFrame,
    x: str,
    bins: List[float] | Tuple[int, float, float],
    weight: str | None = None,
    ax: plt.Axes = None,
    yerr: ArrayLike | bool = True,
    normalise: float | bool = True,
    logbins: bool = False,
    name: str = "",
    title: str = "",
    **kwargs,
) -> Histogram1D:
    """Plot from RDataFrame"""
    if x not in rdf.GetColumnNames():
        raise ValueError(f"No column named {x} in RDataFrame.")

    print(f"Initialising histogram {name}...")

    if logbins:
        if not isinstance(bins, tuple) or (len(bins) != 3):
            raise ValueError(
                "Must pass tuple of (nbins, xmin, xmax) as bins to calculate logarithmic bins"
            )
        bins = np.geomspace(bins[1], bins[2], bins[0] - 1)

    # handle weight
    if weight:
        fill_args = [x, weight]
    else:
        fill_args = [x]

    th1 = ROOT.TH1F(name, title, *get_TH1_bins(bins))
    th1 = rdf.Fill(th1, fill_args).GetPtr()

    # convert to boost
    hist = Histogram1D(th1=th1)
    hist = hist.plot(ax=ax, yerr=yerr, normalise=normalise, **kwargs)

    return hist


def plot(
    df: ROOT.RDataFrame,
    x: str,
    bins: List[float] | Tuple[int, float, float],
    weight: str | None = None,
    ax: plt.Axes = None,
    yerr: ArrayLike | bool = True,
    normalise: float | bool = True,
    logbins: bool = False,
    name: str = "",
    title: str = "",
    show: bool = False,
    name_prefix: str = "",
    name_suffix: str = "",
    logy: bool = False,
    **kwargs,
) -> Histogram1D:
    # naming template for file/histogram name
    if not name:
        name_template = (
            ((name_prefix + "_") if name_prefix else "")  # prefix
            + "{variable}"  # name of variable(s)
            + ("_NORMED" if normalise else "")  # normalisation flag
            + (("_" + name_suffix) if name_suffix else "")  # suffix
        )
        name = name_template.format(variable=x)

    if not ax:
        fig, ax = plt.subplots()

    hist = plot_rdf(
        rdf=df,
        x=x,
        bins=bins,
        weight=weight,
        ax=ax,
        yerr=yerr,
        normalise=normalise,
        logbins=logbins,
        name=name,
        title=title,
        **kwargs,
    )
    set_axis_options(ax, x, bins, lepton="tau", title=title, logx=logbins, logy=logy)

    if show:
        plt.show()

    return hist


if __name__ == "__main__":
    # DTA data
    necessary_variables = {"weight", "mcWeight", "mcChannel", "runNumber", "eventNumber"}
    in_path = "/data/DTA_outputs/2023-01-31/user.kghorban.Sh_2211_Wtaunu_*/*.root"
    trees = {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"}
    cutfile_path = "../../options/DTA_cuts/dta_truth.txt"
    label = r"Sherpa 2211 $W\rightarrow\tau\nu$"
    lumi = 32988.1 + 3219.56

    print("Setting up cutfile..")
    cutfile = Cutfile(cutfile_path, trees)

    # check all branches being extracted from each tree are the same, or TChain will throw a fit
    print("Checking import columns...")
    if all(not x == next(iter(cutfile.tree_dict.values())) for x in cutfile.tree_dict.values()):
        raise ValueError(
            "Can only extract branches with the same name from multiple trees! "
            "Trying to extract the following branches for each tree:\n\n"
            + "\n".join(
                tree + ":\n\t" + "\n\t".join(cutfile.tree_dict[tree]) for tree in cutfile.tree_dict
            )
        )
    import_cols = set(cutfile.tree_dict[next(iter(cutfile.tree_dict))]) | necessary_variables

    # create c++ map for dataset ID metadatas
    # TODO: what's with this tree??
    print("Obtaining metadata...")
    dsid_metadata = ROOT_utils.get_dsid_values(in_path, "T_s1thv_NOMINAL")
    ROOT.gInterpreter.Declare(
        f"""
            std::map<int, float> dsid_sumw{{{','.join(f'{{{t.Index}, {t.sumOfWeights}}}' for t in dsid_metadata.itertuples())}}};
            std::map<int, float> dsid_xsec{{{','.join(f'{{{t.Index}, {t.cross_section}}}' for t in dsid_metadata.itertuples())}}};
            std::map<int, float> dsid_pmgf{{{','.join(f'{{{t.Index}, {t.PMG_factor}}}' for t in dsid_metadata.itertuples())}}};
        """
    )

    print("Making RDataFrame...")
    Rdf = ROOT_utils.init_rdataframe("test_dataframe", in_path, trees)

    # check columns exist in dataframe
    if missing_cols := (set(import_cols) - set(Rdf.GetColumnNames())):
        raise ValueError("Missing column(s) in RDataFrame: \n\t" + "\n\t".join(missing_cols))

    # create weights
    Rdf = (
        Rdf.Define(
            "truth_weight",
            f"(mcWeight * rwCorr * {lumi} * prwWeight * dsid_pmgf[mcChannel]) / dsid_sumw[mcChannel]",
        )
        .Define(
            "base_weight",
            f"(mcWeight * dsid_xsec[mcChannel]) / dsid_sumw[mcChannel]",
        )
        .Define(
            "reco_weight",
            f"(weight * {lumi} * dsid_pmgf[mcChannel]) / dsid_sumw[mcChannel]",
        )
        .Define(
            "ele_reco_weight",
            "reco_weight * Ele_recoSF * Ele_idSF * Ele_isoSF",
        )
        .Define(
            "muon_reco_weight",
            "reco_weight * Muon_recoSF * Muon_isoSF * Muon_ttvaSF",
        )
    )
    weight_cols = {  # MUST change this if weights are changed
        "truth_weight",
        "base_weight",
        "reco_weight",
        "ele_reco_weight",
        "muon_reco_weight",
    }

    # rescale energy columns to GeV
    for gev_column in [
        column
        for column in import_cols
        if (column in variable_data) and (variable_data[column]["units"] == "GeV")
    ]:
        Rdf = Rdf.Redefine(gev_column, f"{gev_column} / 1000")

    # routine to separate vector branches into separate variables
    badcols = set()  # save old vector column names to avoid extracting them later
    print("Shrinking vectors:")
    for col_name in import_cols | weight_cols:
        # unravel vector-type columns
        col_type = Rdf.GetColumnType(col_name)
        debug_str = f"\t- {col_name}: {col_type}"
        if "ROOT::VecOps::RVec" in col_type:
            # skip non-numeric vector types
            if col_type == "ROOT::VecOps::RVec<string>":
                print(f"\t- Skipping string vector column {col_name}")
                badcols.add(col_name)

            elif "jet" in str(col_name).lower():
                # create three new columns for each possible jet
                debug_str += " -> "
                for i in range(3):
                    new_name = col_name + str(i + 1)
                    Rdf = Rdf.Define(f"{new_name}", f"getVecVal(&{col_name},{i})")
                    debug_str += f"{new_name}: {Rdf.GetColumnType(new_name)}, "
                    import_cols.add(new_name)
                badcols.add(col_name)

            else:
                Rdf = Rdf.Redefine(
                    f"{col_name}", f"((&{col_name})->size() > 0) ? (&{col_name})->at(0) : NAN;"
                )
                debug_str += f" -> {Rdf.GetColumnType(col_name)}"
        print(debug_str)

    # calculate derived variables
    print("calculating variables...")
    for derived_var in cutfile.vars_to_calc:
        function = derived_vars[derived_var]["cfunc"]
        args = derived_vars[derived_var]["var_args"]
        func_str = f"{function}({','.join(args)})"

        Rdf = Rdf.Define(derived_var, func_str)

    # create cut columns
    print("Creating cuts...")
    for cut in cutfile.cuts.values():
        Rdf = Rdf.Define(("PASS_" + cut.name), cut.cutstr)

    # PLOTTING
    # ========================================================================================

    # truth
    truth_mass_args = {
        "bins": (30, 1, 5000),
        "logbins": True,
        "logy": True,
    }
    weighted_args = {
        "weight": "truth_weight",
        "title": "truth - 36.2fb$^{-1}$",
        "normalise": False,
    }

    plot(
        Rdf,
        x="TruthMTW",
        **truth_mass_args,
        **weighted_args,
        show=True,
    )
    plot(
        Rdf,
        x="TruthBosonM",
        **truth_mass_args,
        **weighted_args,
        show=True,
    )
    plot(
        Rdf,
        x="TruthTauPt",
        **truth_mass_args,
        **weighted_args,
        show=True,
    )
    plot(
        Rdf,
        x="TruthTauEta",
        bins=(30, -5, 5),
        **weighted_args,
        show=True,
    )
    plot(
        Rdf,
        x="TruthTauPhi",
        bins=(30, -np.pi, np.pi),
        **weighted_args,
        show=True,
    )
