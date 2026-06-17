from __future__ import annotations

import argparse
from pathlib import Path

import ROOT
from tabulate import tabulate

NOMINAL_NAME = "T_s1thv_NOMINAL"
DEFAULT_VARIABLES = ("MTW", "TauPt")
WORKING_POINTS = ("loose", "medium", "tight")
SECTIONS = ("", "1prong_", "3prong_", "tauplus_", "tauminus_")
REGIONS = ("CR_passID", "CR_failID", "SR_passID", "SR_failID")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_input_file() -> Path:
    return repo_root() / "outputs" / "analysis_fakes_2017" / "root" / "wtaunu_lep.root"


def get_hist(file: ROOT.TFile, selection: str, variable: str) -> ROOT.TH1 | None:
    obj = file.Get(f"{NOMINAL_NAME}/{selection}/{variable}")
    if not obj:
        return None
    obj.SetDirectory(0)
    return obj


def integral_with_flow(hist: ROOT.TH1) -> float:
    return hist.Integral(0, hist.GetNbinsX() + 1)


def rows_for_variable(file: ROOT.TFile, variable: str) -> list[list[str]]:
    rows = []
    for wp in WORKING_POINTS:
        for section in SECTIONS:
            for region in REGIONS:
                selection = f"{section}{wp}_{region}"
                true_selection = f"trueTau_{selection}"

                nominal_hist = get_hist(file, selection, variable)
                true_hist = get_hist(file, true_selection, variable)
                if nominal_hist is None or true_hist is None:
                    rows.append(
                        [
                            wp,
                            section.removesuffix("_") or "inclusive",
                            region,
                            "missing",
                            "missing",
                            "missing",
                            "missing",
                        ]
                    )
                    continue

                nominal = integral_with_flow(nominal_hist)
                true_tau = integral_with_flow(true_hist)
                fake_like = nominal - true_tau
                fake_like_percent = 100 * fake_like / nominal if nominal else 0.0

                rows.append(
                    [
                        wp,
                        section.removesuffix("_") or "inclusive",
                        region,
                        f"{nominal:.6g}",
                        f"{true_tau:.6g}",
                        f"{fake_like:.6g}",
                        f"{fake_like_percent:.3g}",
                    ]
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the explicit wtaunu_lep background leaks into the "
            "non-trueTau side used by the fake-factor estimate."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_file(),
        help="Path to analysis_fakes_2017/root/wtaunu_lep.root.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_VARIABLES),
        help="Histogram variables to integrate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(
            f"No wtaunu_lep ROOT file found at {args.input}.\n"
            "Regenerate it first with:\n"
            "  pixi run python run/2017/analysis_fakes_2017.py"
        )

    with ROOT.TFile(str(args.input)) as file:
        if file.IsZombie():
            raise OSError(f"Could not open ROOT file: {args.input}")

        print(f"Input: {args.input}")
        print("fake-like = nominal - trueTau")
        print()
        for variable in args.variables:
            print(f"Variable: {variable}")
            print(
                tabulate(
                    rows_for_variable(file, variable),
                    headers=[
                        "WP",
                        "Section",
                        "Region",
                        "Nominal",
                        "trueTau",
                        "fake-like",
                        "fake-like [%]",
                    ],
                    tablefmt="github",
                )
            )
            print()


if __name__ == "__main__":
    main()
