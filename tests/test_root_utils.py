import numpy as np
import ROOT

from utils import ROOT_utils


def make_hist(name: str, values: list[float]) -> ROOT.TH1F:
    hist = ROOT.TH1F(name, name, len(values), 0, len(values))
    hist.SetDirectory(0)
    for bin_i, value in enumerate(values, start=1):
        hist.SetBinContent(bin_i, value)
    return hist


def test_th1_max_abs_deviation_uses_largest_deviation_per_bin():
    nominal = make_hist("nominal", [100, 100])
    up = make_hist("up", [110, 80])
    down = make_hist("down", [-90, 95])

    uncertainty = ROOT_utils.th1_max_abs_deviation(up, down, nominal)

    np.testing.assert_allclose(ROOT_utils.get_th1_bin_values(uncertainty), [190, 20])


def test_th1_relative_uncertainty_uses_absolute_nominal_and_handles_zero():
    nominal = make_hist("nominal", [100, -50, 0])
    uncertainty = make_hist("uncertainty", [10, 5, 1])

    relative = ROOT_utils.th1_relative_uncertainty(uncertainty, nominal)

    np.testing.assert_allclose(ROOT_utils.get_th1_bin_values(relative), [10, 10, 0])
