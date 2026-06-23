from __future__ import annotations

import ROOT

from utils import ROOT_utils
from utils.ROOT_utils import sum_th1s


def _book_weighted_histogram(
    analysis,
    *,
    dataset: str,
    variable: str,
    selection: str,
    systematic: str,
    hist_name: str,
):
    """Book one weighted histogram without triggering the ROOT event loop."""
    ds = analysis[dataset]
    hist = ds.define_th1(
        variable=variable,
        name=hist_name,
        title=hist_name,
        selection=selection,
    )
    weight = ds._match_weight(variable)
    fill_cols = [variable, weight] if weight else [variable]
    return ds.filters[systematic][selection].df.Fill(hist, fill_cols)


def build_fake_factor_batched(
    analysis,
    *,
    fakes_source: str,
    target_vars: tuple[str, ...],
    cr_pass_selection: str,
    cr_fail_selection: str,
    sr_pass_selection: str,
    sr_fail_selection: str,
    true_cr_pass_selection: str,
    true_cr_fail_selection: str,
    true_sr_fail_selection: str,
    output_prefix: str,
    systematic: str,
    save_intermediates: bool = False,
) -> ROOT.TH1:
    """Build a fake factor while batching the source histograms per dataset.

    The standard Analysis.do_fakes_estimate path asks for each source histogram
    through get_hist(..., allow_generation=True), which immediately evaluates
    each histogram. Here we book the CR/SR source histograms first, then collect
    the results, so ROOT can evaluate the booked actions together for each
    dataset graph.
    """
    if not analysis.data_sample:
        raise ValueError("Cannot build a data-driven fake estimate without a data sample.")

    analysis.logger.info(
        "Calculating batched fake factor for %s with name: '%s'...",
        fakes_source,
        output_prefix,
    )

    prefix = f"{output_prefix}_" if output_prefix else ""
    data_ptrs = {
        "cr_pass": _book_weighted_histogram(
            analysis,
            dataset=analysis.data_sample,
            variable=fakes_source,
            selection=cr_pass_selection,
            systematic=systematic,
            hist_name=f"{prefix}{fakes_source}_{cr_pass_selection}_data",
        ),
        "cr_fail": _book_weighted_histogram(
            analysis,
            dataset=analysis.data_sample,
            variable=fakes_source,
            selection=cr_fail_selection,
            systematic=systematic,
            hist_name=f"{prefix}{fakes_source}_{cr_fail_selection}_data",
        ),
        "sr_fail": _book_weighted_histogram(
            analysis,
            dataset=analysis.data_sample,
            variable=fakes_source,
            selection=sr_fail_selection,
            systematic=systematic,
            hist_name=f"{prefix}{fakes_source}_{sr_fail_selection}_data",
        ),
    }

    mc_ptrs = {
        "cr_pass": [],
        "cr_fail": [],
        "sr_fail": [],
    }
    for mc_sample in analysis.mc_samples:
        mc_ptrs["cr_pass"].append(
            _book_weighted_histogram(
                analysis,
                dataset=mc_sample,
                variable=fakes_source,
                selection=true_cr_pass_selection,
                systematic=systematic,
                hist_name=f"{prefix}{fakes_source}_{true_cr_pass_selection}_{mc_sample}",
            )
        )
        mc_ptrs["cr_fail"].append(
            _book_weighted_histogram(
                analysis,
                dataset=mc_sample,
                variable=fakes_source,
                selection=true_cr_fail_selection,
                systematic=systematic,
                hist_name=f"{prefix}{fakes_source}_{true_cr_fail_selection}_{mc_sample}",
            )
        )
        mc_ptrs["sr_fail"].append(
            _book_weighted_histogram(
                analysis,
                dataset=mc_sample,
                variable=fakes_source,
                selection=true_sr_fail_selection,
                systematic=systematic,
                hist_name=f"{prefix}{fakes_source}_{true_sr_fail_selection}_{mc_sample}",
            )
        )

    h_cr_pass_data = data_ptrs["cr_pass"].GetValue()
    h_cr_fail_data = data_ptrs["cr_fail"].GetValue()
    h_sr_fail_data = data_ptrs["sr_fail"].GetValue()
    h_cr_pass_mc = sum_th1s(*[ptr.GetValue() for ptr in mc_ptrs["cr_pass"]])
    h_cr_fail_mc = sum_th1s(*[ptr.GetValue() for ptr in mc_ptrs["cr_fail"]])
    h_sr_fail_mc = sum_th1s(*[ptr.GetValue() for ptr in mc_ptrs["sr_fail"]])

    numerator = h_cr_pass_data - h_cr_pass_mc
    denominator = h_cr_fail_data - h_cr_fail_mc
    fake_data_estimate = h_sr_fail_data - h_sr_fail_mc

    fake_factor = numerator / denominator
    fake_factor.SetName(f"{prefix}{fakes_source}_FF")
    fake_factor.SetTitle(f"{prefix}{fakes_source}_FF")
    fake_factor.SetDirectory(0)

    source_prediction = fake_data_estimate * fake_factor
    source_prediction.SetName(f"{prefix}{fakes_source}_fakes_bkg_{fakes_source}")
    source_prediction.SetTitle(f"{prefix}{fakes_source}_fakes_bkg_{fakes_source}")
    source_prediction.SetDirectory(0)

    analysis.histograms[fake_factor.GetName()] = fake_factor
    analysis.histograms[source_prediction.GetName()] = source_prediction

    if save_intermediates:
        analysis.histograms[f"{prefix}all_mc_{fakes_source}_{true_cr_pass_selection}"] = (
            h_cr_pass_mc
        )
        analysis.histograms[f"{prefix}all_mc_{fakes_source}_{true_cr_fail_selection}"] = (
            h_cr_fail_mc
        )
        analysis.histograms[f"{prefix}all_mc_{fakes_source}_{true_sr_fail_selection}"] = (
            h_sr_fail_mc
        )
        analysis.histograms[f"{prefix}{fakes_source}_FF_numerator"] = numerator
        analysis.histograms[f"{prefix}{fakes_source}_FF_denominator"] = denominator
        analysis.histograms[f"{prefix}{fakes_source}_FF_fakes_data_est"] = fake_data_estimate

    fill_fake_predictions_from_factor(
        analysis,
        target_vars=target_vars,
        sr_pass_selection=sr_pass_selection,
        sr_fail_selection=sr_fail_selection,
        true_sr_fail_selection=true_sr_fail_selection,
        ff_hist=fake_factor,
        output_prefix=output_prefix,
        fakes_source=fakes_source,
        systematic=systematic,
    )
    analysis.logger.info("Completed batched fake estimate for '%s'", output_prefix)
    return fake_factor


def fill_fake_predictions_from_factor(
    analysis,
    *,
    target_vars: tuple[str, ...],
    sr_pass_selection: str,
    sr_fail_selection: str,
    true_sr_fail_selection: str,
    ff_hist: ROOT.TH1,
    output_prefix: str,
    fakes_source: str,
    systematic: str,
) -> None:
    """Apply an already-derived fake factor to one SR fail-ID region."""
    ff_symbol = f"FF_hist_{output_prefix or 'default'}_{fakes_source}"
    ROOT.gInterpreter.Declare(
        f"TH1* {ff_symbol} = reinterpret_cast<TH1*>({ROOT.addressof(ff_hist)});"
    )
    ff_weight_col = f"FF_weight_{output_prefix}_{fakes_source}"
    ff_weight = (
        f"reco_weight"
        f" * {ff_symbol}->GetBinContent("
        f"{ff_symbol}->FindBin({fakes_source}))"
    )

    data_df = (
        analysis[analysis.data_sample]
        .filters[systematic][sr_fail_selection]
        .df.Define(ff_weight_col, ff_weight)
    )
    mc_dfs = {
        mc_sample: analysis[mc_sample]
        .filters[systematic][true_sr_fail_selection]
        .df.Define(ff_weight_col, ff_weight)
        for mc_sample in analysis.mc_samples
    }

    ff_hists = {}
    for target_var in target_vars:
        hist_name = f"{output_prefix}_{target_var}_fakes_bkg_{fakes_source}_src"
        h_bins = analysis[analysis.data_sample].get_binnings(target_var, sr_pass_selection)
        data_hist = ROOT.TH1F(
            f"{hist_name}_data",
            hist_name,
            *ROOT_utils.get_TH1_bin_args(**h_bins),
        )
        data_ptr = data_df.Fill(data_hist, [target_var, ff_weight_col])

        mc_ptrs = []
        for mc_sample, mc_df in mc_dfs.items():
            mc_hist = ROOT.TH1F(
                f"{hist_name}_{mc_sample}",
                hist_name,
                *ROOT_utils.get_TH1_bin_args(**h_bins),
            )
            mc_ptrs.append(mc_df.Fill(mc_hist, [target_var, ff_weight_col]))
        ff_hists[target_var] = (hist_name, data_ptr, mc_ptrs)

    for target_var, (hist_name, data_ptr, mc_ptrs) in ff_hists.items():
        analysis.logger.info(
            "Calculating fake background estimate for %s with cached FF name: '%s'...",
            target_var,
            output_prefix,
        )
        fake_prediction = data_ptr.GetValue() - sum_th1s(*[ptr.GetValue() for ptr in mc_ptrs])
        fake_prediction.SetName(hist_name)
        fake_prediction.SetTitle(hist_name)
        fake_prediction.SetDirectory(0)
        for bin_idx in range(1, fake_prediction.GetNbinsX() + 1):
            fake_prediction.SetBinError(
                bin_idx,
                fake_prediction.GetBinContent(bin_idx) * 0.1,
            )
        analysis.histograms[hist_name] = fake_prediction
