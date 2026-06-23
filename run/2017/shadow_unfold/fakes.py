import ROOT
from samples import NOMINAL_NAME

from src.analysis import Analysis
from utils import ROOT_utils
from utils.ROOT_utils import sum_th1s


def fake_like_histogram(
    analysis: Analysis,
    variable: str,
    selection: str,
    *,
    name: str,
) -> ROOT.TH1:
    """Return data minus MC contamination for one validation selection."""
    data_hist = analysis.get_hist(
        variable,
        dataset=analysis.data_sample,
        systematic=NOMINAL_NAME,
        selection=selection,
        allow_generation=True,
    )
    mc_contamination_hist = sum_th1s(
        *[
            analysis.get_hist(
                variable,
                dataset=mc_sample,
                systematic=NOMINAL_NAME,
                selection=f"trueTau_{selection}",
                allow_generation=True,
            )
            for mc_sample in analysis.mc_samples
        ]
    )
    fake_like = data_hist - mc_contamination_hist
    fake_like.SetName(name)
    fake_like.SetDirectory(0)
    return fake_like


def positive_unit_shape(hist: ROOT.TH1, name: str) -> ROOT.TH1:
    """Return a positive unit-normalised shape for transfer reweighting."""
    shape = hist.Clone(name)
    shape.SetDirectory(0)
    for bin_idx in range(1, shape.GetNbinsX() + 1):
        if shape.GetBinContent(bin_idx) < 0:
            shape.SetBinContent(bin_idx, 0.0)
            shape.SetBinError(bin_idx, 0.0)
    integral = shape.Integral()
    if integral > 0:
        shape.Scale(1.0 / integral)
    return shape


def shape_ratio_histogram(
    numerator: ROOT.TH1,
    denominator: ROOT.TH1,
    name: str,
) -> ROOT.TH1:
    """Build a bin-by-bin ratio between two unit-normalised shapes."""
    ratio_hist = numerator.Clone(name)
    ratio_hist.SetDirectory(0)
    for bin_idx in range(1, ratio_hist.GetNbinsX() + 1):
        denominator_value = denominator.GetBinContent(bin_idx)
        if denominator_value <= 0:
            ratio_hist.SetBinContent(bin_idx, 0.0)
        else:
            ratio_hist.SetBinContent(
                bin_idx,
                numerator.GetBinContent(bin_idx) / denominator_value,
            )
        ratio_hist.SetBinError(bin_idx, 0.0)
    return ratio_hist


def fill_width_reweighted_fake_prediction_from_factor(
    analysis: Analysis,
    *,
    target_var: str,
    sr_pass_selection: str,
    sr_fail_selection: str,
    true_sr_fail_selection: str,
    ff_hist: ROOT.TH1,
    width_ratio_hist: ROOT.TH1,
    width_variable: str,
    output_name: str,
    fakes_source: str,
) -> ROOT.TH1:
    """Apply a fake factor and a tau-width transfer weight to one fail-ID region."""
    ROOT.gInterpreter.Declare(
        f"TH1* FF_hist_{output_name} = reinterpret_cast<TH1*>({ROOT.addressof(ff_hist)});"
    )
    ROOT.gInterpreter.Declare(
        f"TH1* width_ratio_{output_name} = "
        f"reinterpret_cast<TH1*>({ROOT.addressof(width_ratio_hist)});"
    )

    weight_col = f"FF_width_weight_{output_name}"
    weight_expr = (
        f"reco_weight"
        f" * FF_hist_{output_name}->GetBinContent("
        f"FF_hist_{output_name}->FindBin({fakes_source}))"
        f" * width_ratio_{output_name}->GetBinContent("
        f"width_ratio_{output_name}->FindBin({width_variable}))"
    )

    h_bins = analysis[analysis.data_sample].get_binnings(target_var, sr_pass_selection)
    data_hist = ROOT.TH1F(
        f"{output_name}_data",
        output_name,
        *ROOT_utils.get_TH1_bin_args(**h_bins),
    )
    data_ptr = (
        analysis[analysis.data_sample]
        .filters[NOMINAL_NAME][sr_fail_selection]
        .df.Define(weight_col, weight_expr)
        .Fill(data_hist, [target_var, weight_col])
    )

    mc_ptrs = []
    for mc_sample in analysis.mc_samples:
        mc_hist = ROOT.TH1F(
            f"{output_name}_{mc_sample}",
            output_name,
            *ROOT_utils.get_TH1_bin_args(**h_bins),
        )
        mc_ptrs.append(
            analysis[mc_sample]
            .filters[NOMINAL_NAME][true_sr_fail_selection]
            .df.Define(weight_col, weight_expr)
            .Fill(mc_hist, [target_var, weight_col])
        )

    fake_prediction = data_ptr.GetValue() - sum_th1s(*[ptr.GetValue() for ptr in mc_ptrs])
    fake_prediction.SetName(output_name)
    fake_prediction.SetTitle(output_name)
    fake_prediction.SetDirectory(0)
    for bin_idx in range(1, fake_prediction.GetNbinsX() + 1):
        fake_prediction.SetBinError(
            bin_idx,
            abs(fake_prediction.GetBinContent(bin_idx)) * 0.1,
        )
    analysis.histograms[output_name] = fake_prediction
    return fake_prediction
