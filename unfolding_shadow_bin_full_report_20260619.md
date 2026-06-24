# Shadow-Bin Unfolding Closure Report

Initial check: 2026-06-19

Updated: 2026-06-24

## Executive Summary

The current closure study shows that the shadow-bin unfolding now closes for signal MC once the reconstructed nonfiducial signal contribution is removed before unfolding.

The key result is:

- `MTW` signal-MC closure is at the per-mille level for no-shadow and all MTW shadow-bin configurations tested.
- `TauPt` signal-MC closure was also demonstrated at the per-mille level during the diagnostic phase, but the cleaned production script now runs `MTW` only because that is the measured observable.
- The earlier 10-30% closure failure was not caused primarily by the shadow-bin threshold itself. It was caused by unfolding reconstructed signal events that pass the reco selection but do not belong to the nominal truth fiducial phase space.
- The latest run also fixes the reconstructed background bookkeeping: the nominal unfolded input now subtracts MC-contamination backgrounds plus data-driven jet fakes, rather than subtracting all MC backgrounds plus data-driven fakes. The old all-MC convention is retained only as a diagnostic row.

The current implementation is therefore a defensible fiducial unfolding treatment:

1. Build the measured input in the same reconstructed phase space as the response.
2. Subtract MC-contamination backgrounds and data-driven jet-fake estimates.
3. Subtract reconstructed `wtaunu_had` events that fail the nominal truth fiducial definition as a nonfiducial signal contribution.
4. Unfold the remaining input with a response built from nominal truth-fiducial signal.

The latest completed cleaned production run is:

`outputs/analysis_shadow_unfold/logs/analysis_shadow_unfold_2026-06-24_14-22-21.log`

That log ends with `DONE.` and wrote the current cleaned `MTW` closure summary. It has the high-priority fake-source systematics enabled:

- `JET_FAKE_FF_STAT`;
- `JET_FAKE_MET_WINDOW`;
- `JET_FAKE_TAU_WIDTH_COMPOSITION`.

It also attempts the response-systematic pass. The response cache was correctly rebuilt after stale cutflows were detected, but the TES response variations are still skipped because the full up/down response objects are unavailable. Earlier fake-diagnostic and split-closure sections in this report remain based on the dedicated validation runs listed in their sections.

The previous all-MC-background output was archived before this bookkeeping change:

`outputs/analysis_shadow_unfold_baseline_allmc_background_20260623/`

## Terminology

The report now follows the terminology used in ATLAS fake-factor discussions.
Earlier sections and some cached histogram names still use the historical
framework prefix `trueTau_...`; that prefix should be read as an implementation
label, not as the fiducial signal truth definition.

Preferred wording:

| This report/code now says | Meaning | Older shorthand to avoid in prose |
|---|---|---|
| jet-fake estimate | data-driven estimate for jets misidentified as `tau_had-vis` | generic "fakes" when the source matters |
| MC-contamination subtraction | simulated real-object/non-jet contamination subtracted before forming or applying fake factors | nonfake MC |
| fake-like or jet-fake-like residual | `data - MC contamination`, intended to be dominated by jet-to-tau fakes | truth / true tau |
| ID subregion | tau candidate passes the analysis tau-ID requirement | pass-ID |
| anti-ID subregion | tau candidate fails the analysis tau-ID but passes the anti-ID floor | fail-ID |
| determination region | region where fake factors are measured | fake CR, unless referring to code names |

In the current implementation, the MC-contamination subtraction contains
matched hadronic tau, muon, and electron candidates. Photon-matched candidates
remain a commented validation/systematic variant, not the nominal definition.

## Scope Of The Current Closure Test

The current closure workflow is:

`run/2017/analysis_shadow_unfold.py`

It writes to:

`outputs/analysis_shadow_unfold`

After the 2026-06-23 cleanup, the production script is intentionally narrow:

- year: `2017`
- tau ID: `medium`
- prongs: inclusive `1+3`, with prong-split fake factors
- variable: `MTW`
- systematics: enabled for the latest smoke run, `DO_FULL_SYSTEMATICS = True`
- 1-prong tau-width fake-source shift: enabled, `RUN_FAKE_WIDTH_SYSTEMATIC = True`
- fake-factor source: `TauPt`
- fake control region: `TauPt > 170` and `MET_met < 100`
- fake diagnostics: moved to `run/2017/validations/`
- split-sample closure: moved to `run/2017/validations/validate_split_sample_unfolding_closure.py`
- shadow thresholds:
  - current run: `MTW` no-shadow and `250` GeV lower reconstructed threshold
  - earlier diagnostic runs: `MTW` `200`, `250`, and `300` GeV lower reconstructed thresholds

The current script no longer includes the temporary `MTW_MET_category_shadow_bin_250` diagnostic. That category test was useful for diagnosing MET-related reco failures, but it is not part of the current analysis path because the nonfiducial signal correction addresses the actual fiducial mismatch more cleanly.

The latest completed output directory still contains some stale diagnostic category plots from the previous run. They should be ignored unless explicitly discussed as a historical diagnostic.

## Current Closure Results

Summary file:

`outputs/analysis_shadow_unfold/closure_summary.md`

Relevant rows from the latest cleaned production run:

| Configuration | Variable | Iterations | Mean deviation | Max deviation | Integral ratio |
|---|---|---:|---:|---:|---:|
| no_shadow_bin | MTW | 0 | 0.000 | 0.000 | 1.000 |
| no_shadow_bin | MTW | 1 | 0.001 | 0.007 | 1.000 |
| no_shadow_bin | MTW | 2 | 0.001 | 0.010 | 1.000 |
| MTW_shadow_bin_250 | MTW | 0 | 0.000 | 0.000 | 1.000 |
| MTW_shadow_bin_250 | MTW | 1 | 0.001 | 0.007 | 1.000 |
| MTW_shadow_bin_250 | MTW | 2 | 0.001 | 0.010 | 1.000 |

Earlier pre-cleanup diagnostic runs also demonstrated the same per-mille `MTW` closure for the `200` and `300` GeV shadow thresholds and for diagnostic `TauPt` configurations. Those rows are retained in the historical validation sections below, but they are not produced by the current cleaned production configuration.

This is a decisive improvement over the pre-correction variable-specific run, where Bayesian-unfolded signal MC overshot truth by roughly 9-15% in integral, and over the earlier full-shadow-response attempt, where the response and measured input were defined in inconsistent reconstructed phase spaces.

## MC-Contamination Background Bookkeeping Correction

Question:
Should the data-driven fake estimate be added on top of all reconstructed MC
backgrounds, or should it replace the jet-fake-like part of the reconstructed MC
background prediction?

Implementation:
- script: `run/2017/analysis_shadow_unfold.py`
- archived baseline: `outputs/analysis_shadow_unfold_baseline_allmc_background_20260623/`
- corrected output: `outputs/analysis_shadow_unfold/`
- summary files:
  - baseline: `outputs/analysis_shadow_unfold_baseline_allmc_background_20260623/closure_summary.md`
  - corrected: `outputs/analysis_shadow_unfold/closure_summary.md`
- mode: production-style central run, using cached measured/response histograms where possible.

The corrected nominal input is now:

```text
data_sig = data
         - MC-contamination backgrounds
         - data-driven fakes
         - nonfiducial signal
```

The old convention is still written in the corrected summary as:

```text
Data sig, all bkg + fakes diagnostic
```

This diagnostic reproduces the archived baseline values, so the before/after
comparison is controlled.

| Configuration | Old data sig, all MC bkg + fakes | Corrected data sig, MC-contam bkg + fakes | Jet-fake-like MC bkg restored | Fid reco signal | Fid reco / old data sig | Fid reco / corrected data sig | Fid reco / width-shifted corrected data sig |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_shadow_bin` | `661.192` | `774.800` | `113.607` | `896.196` | `1.355` | `1.157` | `1.096` |
| `MTW_shadow_bin_250` | `675.028` | `799.343` | `124.315` | `934.116` | `1.384` | `1.169` | `1.106` |

The corrected pre-unfolding budget is:

| Configuration | Data | All MC bkg | MC-contam bkg | Jet-fake-like MC bkg | Fakes | Nonfid signal | Corrected data sig |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_shadow_bin` | `1351.000` | `359.137` | `245.530` | `113.607` | `226.136` | `104.534` | `774.800` |
| `MTW_shadow_bin_250` | `1428.000` | `388.381` | `264.066` | `124.315` | `243.815` | `120.776` | `799.343` |

Representative corrected plots:
- `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/no_shadow_bin_MTW_1iter_unfolded.png`
- `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/MTW_shadow_bin_250_MTW_1iter_unfolded.png`
- `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/fake_width_systematic/no_shadow_bin_MTW_TauTrackWidthPt1000PV_1iter_fake_width_shift.png`
- `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/fake_width_systematic/MTW_shadow_bin_250_MTW_TauTrackWidthPt1000PV_1iter_fake_width_shift.png`

Interpretation:
the bookkeeping correction is material. It removes the double subtraction of
jet-fake-like reconstructed MC and improves the fiducial reco signal over
background-subtracted data ratio from `1.355` to `1.157` for no-shadow and from
`1.384` to `1.169` for `MTW_shadow_bin_250`. The validated 1-prong tau-width
fake-source shift moves the ratios further to `1.096` and `1.106`.

This is the most coherent result so far: the old all-MC subtraction was too
aggressive once data-driven fakes were also applied. The corrected convention is
closer to the fake-factor logic used in ATLAS analyses, where simulated
real-object/non-jet contamination is subtracted and the data-driven fake estimate represents the
jet-fake component.

Recommendation:
use the MC-contamination-background bookkeeping as the nominal central-value convention
for this shadow-unfolding workflow. Keep the photon-expanded contamination definition
as a commented validation/systematic variant until a dedicated comparison is
adopted. The remaining `10-17%` normalisation tension should be treated as a
fake-modelling/systematic question, not as an unfolding-response closure
failure.

## Nonfiducial Signal Correction

The correction applied in the current script is:

```python
all_reco_signal = response_analysis.get_hist(var, "wtaunu_had", selection=reco_selection)
fiducial_reco_signal = response_analysis.get_hist(var, "wtaunu_had", selection=truth_reco_selection)
nonfiducial_signal = all_reco_signal - fiducial_reco_signal

data_sig = data - background - nonfiducial_signal
signal = fiducial_reco_signal
response = RooUnfoldResponse(fiducial_reco_signal, truth_response, matrix)
```

This means:

- `all_reco_signal`: reconstructed `wtaunu_had` passing the reco selection.
- `fiducial_reco_signal`: reconstructed `wtaunu_had` that also satisfies the nominal truth fiducial definition.
- `nonfiducial_signal`: reconstructed signal outside the fiducial measurement target.

The nonfiducial component is subtracted before unfolding because the unfolded result is intended to represent the nominal truth fiducial phase space.

The correction sizes from the completed run are:

| Configuration | Variable | Nonfiducial fraction of reco signal |
|---|---|---:|
| no_shadow_bin | MTW | 10.4% |
| MTW_shadow_bin_250 | MTW | 11.4% |

These values are large enough to explain the previous nonclosure. They are also physically sensible: relaxing the reconstructed selection admits more reco-selected signal that is not part of the nominal truth fiducial target.

## Current Tau-width Fake-source Shift

The latest cleaned run also propagated the validated `TauTrackWidthPt1000PV`
`application_to_lowmet` width-shape shift through the final `MTW` unfolding. This
is applied only to the 1-prong fake component; the 3-prong component is kept
nominal because the high-MET 3-prong validation target remains negative after
MC-contamination subtraction and is not physically usable as a transfer target.

Summary file:

`outputs/analysis_shadow_unfold/closure_summary.md`

| Configuration | Nominal fakes | Width-shifted fakes | Nominal data sig | Shifted data sig | Fid reco / nominal data sig | Fid reco / shifted data sig |
|---|---:|---:|---:|---:|---:|---:|
| no_shadow_bin | 226.136 | 182.892 | 774.800 | 818.044 | 1.157 | 1.096 |
| MTW_shadow_bin_250 | 243.815 | 198.235 | 799.343 | 844.923 | 1.169 | 1.106 |

The width shift therefore reduces the total fake estimate by about `19%` in the
no-shadow configuration and about `18.7%` in the `MTW` 250 GeV shadow-bin
configuration. It moves the data-derived signal input closer to the fiducial
reco signal expectation, but it does not remove the normalisation tension:

- no-shadow: fiducial reco / data signal improves from `1.157` to `1.096`;
- `MTW_shadow_bin_250`: fiducial reco / data signal improves from `1.169` to
  `1.106`.

Representative current-run plots:

| no-shadow width shift | `MTW` shadow 250 width shift |
|---|---|
| <img src="outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/fake_width_systematic/no_shadow_bin_MTW_TauTrackWidthPt1000PV_1iter_fake_width_shift.png" width="390"> | <img src="outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/fake_width_systematic/MTW_shadow_bin_250_MTW_TauTrackWidthPt1000PV_1iter_fake_width_shift.png" width="390"> |

Interpretation:
the tau-width shift behaves in the expected direction and is large enough to be
a credible 1-prong fake-source composition uncertainty. It should still not be
treated as a nominal correction without a final systematic prescription. The
remaining discrepancy is still too large to attribute entirely to the 1-prong
width-shape transfer; the 3-prong/MC-contamination-subtraction issue remains separate.

## Pre-Unfolding Stack Check

Question:
Before interpreting any unfolded result, does the ordinary reconstructed-level
Data/MC comparison already show the normalisation problem?

Implementation:
- script: `run/2017/analysis_shadow_unfold.py`
- output summary: `outputs/analysis_shadow_unfold/closure_summary.md`
- mode: cache/output inspection only; no new ROOT event loops were run for this
  check.

The current production script does not yet save a conventional SR stack plot
with data overlaid on stacked signal, MC backgrounds, and fake estimate.
However, the current output summary contains the same ingredients. For a
standard pre-unfolding stack, the reconstructed-level prediction is:

```text
prediction = fiducial reco signal + nonfiducial reco signal + MC backgrounds + fakes
```

The table below compares this reconstructed-level prediction to the observed
data. The "no fakes" column is not a proposed nominal model; it is a diagnostic
showing the same stack with only the fake component removed.

| Configuration | Data | Prediction without fakes | Pred./data without fakes | Prediction with nominal fakes | Pred./data with nominal fakes | Prediction with width-shift fakes | Pred./data with width-shift fakes |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_shadow_bin, `MTW` | 1351.000 | 1359.867 | 1.007 | 1586.003 | 1.174 | 1542.759 | 1.142 |
| `MTW_shadow_bin_250`, `MTW` | 1428.000 | 1443.273 | 1.011 | 1687.088 | 1.181 | 1641.508 | 1.150 |

Result:
the pre-unfolding stack already identifies the core issue. Without the fake
component, the reconstructed-level MC prediction agrees with data at about the
`1%` level. Adding the current nominal fake estimate makes the prediction
overshoot data by about `17-18%`. Applying the validated 1-prong tau-width
shape shift reduces this to about `14-15%`, but it does not solve the
normalisation problem.

Interpretation:
the disagreement is visible before unfolding. The unfolding machinery is not
creating the normalisation tension; it is propagating a data-minus-background
input that has already been driven low by the fake subtraction. This reinforces
the conclusion from the fake-scale diagnostics: the immediate analysis problem
is the fake estimate/background composition at reconstructed level, not the
Bayesian unfolding iteration or response closure.

Recommendation:
add explicit pre-unfolding SR stack plots to the production or validation
workflow before doing further unfolding scans. The useful plot set is:

- nominal current fake model;
- no-fake diagnostic overlay;
- 1-prong tau-width shifted fake model;
- optionally the thesis/inclusive fake model for direct comparison.

These plots should be generated from saved histograms where possible. They are
more thesis-useful than another unfolding rerun for diagnosing the current
Data/MC normalisation.

## Current Working Interpretation Of The Fake-Normalisation Problem

Question:
Why does the reconstructed-level Data/MC comparison agree at the `1%` level
without the data-driven fake estimate, but overshoot data once the corrected fake
estimate is added?

This section is a synthesis of the validation studies above. It is not a new
ROOT run.

Observed evidence:

| Observation | Evidence | Current interpretation |
|---|---|---|
| The fake-factor formula was genuinely corrected | commit `5d9535a`; see "Fake-background implementation corrected" below | The post-fix fake estimate should not be rejected merely because it is larger than the thesis-era estimate. The old implementation was not the intended anti-ID transfer. |
| No-fake reconstructed stack agrees with data at about `1%` | `outputs/analysis_shadow_unfold/closure_summary.md` | This is suspiciously good and may be an accidental cancellation. It does not prove the true jet-fake contribution is zero. |
| Adding the current fake estimate overshoots data by `17-18%` | pre-unfolding stack check above | The disagreement is present before unfolding; the Bayesian unfolding is not creating the normalisation tension. |
| Fake-factor transfer works in control-like regions | low-MET and MTW-sideband validation summaries | The fake method is not globally broken. It works where the validation target is well behaved. |
| Fake-factor transfer is strained in high-`MET` signal-like regions | MET-binned and low-MET fake-region validations | The high-`MET` application region has different composition/kinematics from the fake-factor derivation region. |
| The 1-prong tau-width shift moves the fake estimate in the right direction | tau-width composition and reweighting summaries | Tau-width is a credible fake-source composition systematic for 1-prong fakes, not yet a central correction. |
| The 3-prong high-`MET` validation target is negative before fake prediction | MC-contamination-subtraction validation | The 3-prong problem is dominated by the simulated real-object contamination subtraction, especially `wtaunu_had`, not by fake-factor binning alone. |
| `wtaunu_had` is too 3-prong-heavy after weighting | prong-balance diagnostics | The reconstructed MC-contamination subtraction likely has a prong-dependent modelling or weighting issue. |

The current best interpretation is therefore:

1. The corrected fake-factor algebra is likely right.
2. The current high-`MET` fake transfer is not fully reliable, especially for
   1-prong composition and 3-prong validation targets.
3. The 3-prong issue is not primarily a fake-factor issue; the pass-ID
   `data - MC contamination` target is already negative because the simulated
   real-object contamination from
   `wtaunu_had` subtraction is too large.
4. The apparently excellent no-fake Data/MC agreement may be an accidental
   cancellation between missing real jet fakes and an over-large MC-contamination
   prediction.

### Reconstructed Fake-like MC Bookkeeping Check

The reconstructed stack-composition validation and the latest production run now
answer the bookkeeping question directly: adding the data-driven fake estimate
on top of all reconstructed MC double counts fake-like MC. The central
unfolding workflow should therefore subtract MC-contamination backgrounds and the
data-driven fake estimate, while retaining the all-MC subtraction only as a
diagnostic comparison.

The diagnostic stack comparison is:

```text
1. all MC, no data-driven fakes
2. all MC + data-driven fakes
3. MC-contamination component + data-driven fakes
4. jet-fake-like reconstructed MC only
```

The decisive comparison is:

```text
all MC + fakes
vs
MC-contamination component + fakes
```

If removing jet-fake-like reconstructed MC from the stack restores agreement, the
issue is double counting: the data-driven estimate should replace that MC
component rather than be added on top of it. If the stack still overshoots after
jet-fake-like MC is removed, then the fake estimate itself is too large or the
MC-contamination subtraction is still overestimated.

### Truth-Category Audit Still Needed

The report currently has extensive prong-balance and MC-contamination-subtraction
diagnostics, but it does not yet contain a full reconstructed truth-category
audit. The next validation should split the selected MC into categories such as:

- true hadronic tau;
- true electron or muon matched;
- photon matched;
- unmatched or light-jet fake-like;
- other/unknown.

This check matters because the fake-factor method subtracts MC contamination from
the anti-ID data before applying the fake factor. If the truth-category
definition used for that subtraction is too narrow or too broad, the fake
prediction can be biased. In particular, photon-matched tau candidates have not
yet been isolated in the current report.

Recommendation:
the next production-relevant validation should be a pre-unfolding stack
decomposition, not another unfolding scan. It should use the saved histograms
where possible and only run new dataframe loops for truth-category histograms
that are not already cached. A nominal fake-model change should wait until this
double-counting/truth-category question is resolved.

## Representative Plots

The `MTW` 250 GeV shadow-bin result now closes for signal MC. The unfolded signal MC sits on truth MC, while unfolded data remains lower than the signal prediction in several bins. This is good: the correction fixes MC closure without forcing data to agree with signal MC.

<img src="outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/MTW_shadow_bin_250_MTW_1iter_unfolded.png" width="520">

The no-shadow `MTW` result also closes after the same correction:

<img src="outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/no_shadow_bin_MTW_1iter_unfolded.png" width="520">

The cleaned production run no longer emits `TauPt` plots; those are historical diagnostics only. The current `MTW` response matrix remains populated and broadly diagonal:

<img src="outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/MTW_shadow_bin_250_MTW_response_matrix.png" width="520">

## Fake-Estimate Diagnostics From The Diagnostic Run

Summary file:

`outputs/analysis_shadow_unfold/fake_diagnostics_summary.md`

This section records the pre-cleanup diagnostic output. The same questions are now owned by dedicated scripts under `run/2017/validations/`, so this section should be read as historical evidence rather than a description of the current production runner.

The diagnostic run added three fake-estimate diagnostics:

1. a pre-unfolding event budget;
2. a fake-scale scan, where the fake estimate is scaled by `0`, `0.5`, and `1`;
3. MC-only fake closure and inclusive-versus-prong-split fake comparisons.

These diagnostics do not change the nominal unfolded result. They test whether the fake estimate is driving the normalisation difference between unfolded data and signal MC.

### Pre-unfolding budget

The relevant measured input is:

```text
data_sig = data - MC backgrounds - fake estimate - nonfiducial signal
```

For `MTW`, the budget shows a clear pattern:

| Configuration | Data | MC bkg | Fakes | Nonfid signal | Data sig | Data sig, no fakes | Fid reco signal | Fid reco / data sig |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_shadow_bin | 1351.000 | 359.137 | 226.136 | 104.534 | 661.192 | 887.328 | 896.196 | 1.355 |
| MTW_shadow_bin_250 | 1428.000 | 388.381 | 243.815 | 120.776 | 675.028 | 918.843 | 934.116 | 1.384 |

For `TauPt`, the same effect is present but smaller:

| Configuration | Data | MC bkg | Fakes | Nonfid signal | Data sig | Data sig, no fakes | Fid reco signal | Fid reco / data sig |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_shadow_bin | 1351.000 | 359.033 | 152.952 | 104.507 | 734.508 | 887.460 | 896.216 | 1.220 |
| TauPt_shadow_bin_200 | 1420.000 | 375.691 | 152.952 | 82.288 | 809.068 | 962.020 | 943.211 | 1.166 |
| TauPt_shadow_bin_250 | 1420.000 | 375.691 | 152.952 | 83.407 | 807.949 | 960.901 | 942.092 | 1.166 |
| TauPt_shadow_bin_300 | 1417.000 | 375.691 | 152.952 | 85.698 | 802.658 | 955.611 | 939.801 | 1.171 |

The striking point is that `data_sig, no fakes` is already close to the fiducial reconstructed signal expectation. Applying the full fake subtraction moves the data-derived signal input well below the signal-MC expectation.

For example, in `MTW_shadow_bin_250`:

```text
data_sig with full fakes = 675.028
data_sig with no fakes   = 918.843
fiducial reco signal MC  = 934.116
```

This is why the unfolded data lies below signal MC in the diagnostic plots even though the signal-MC closure is good.

### Fake-scale scan

The fake-scale scan makes the same point visually. For `MTW_shadow_bin_250`, the full fake estimate undershoots truth MC, while the zero-fake curve is much closer to truth MC over the populated `MTW` range:

<img src="outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/fake_diagnostics/MTW_shadow_bin_250_MTW_fake_scale_scan.png" width="520">

For `TauPt_shadow_bin_250`, the effect is smaller but still present:

<img src="outputs/analysis_shadow_unfold/plots/TauPt_shadow_bin_250/TauPt/fake_diagnostics/TauPt_shadow_bin_250_TauPt_fake_scale_scan.png" width="520">

This does not prove that the true fake contribution is zero. It shows that the current nominal fake subtraction is large enough to dominate the data/MC normalisation difference in this reduced shadow-unfold workflow.

### MC fake closure

The MC-only fake closure compares:

- the known MC jet-fake-like component passing the signal selection;
- the fake-factor prediction for that same component.

The integral ratios are:

| Configuration | Actual MC fake | Predicted MC fake | Integral ratio |
|---|---:|---:|---:|
| no_shadow_bin | 113.643 | 105.853 | 0.931 |
| MTW_shadow_bin_200 | 124.747 | 98.515 | 0.790 |
| MTW_shadow_bin_250 | 124.351 | 106.191 | 0.854 |
| MTW_shadow_bin_300 | 122.717 | 107.736 | 0.878 |
| TauPt_shadow_bin_200 | 117.231 | 105.853 | 0.903 |
| TauPt_shadow_bin_250 | 117.231 | 105.853 | 0.903 |
| TauPt_shadow_bin_300 | 117.231 | 105.853 | 0.903 |

This is important because it means the issue is not simply that the fake-factor method overpredicts MC fakes. In MC-only closure, the prediction is generally lower than the actual fake component. The data-driven fake estimate can still be too large for the unfolded data input because the data control-region composition, prompt subtraction, or fake-enriched phase space may differ from the MC-only test.

Representative MC fake closure plot:

<img src="outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/fake_diagnostics/MTW_shadow_bin_250_TauPt_mc_fake_closure.png" width="520">

### Inclusive versus prong-split fakes

The thesis fake estimate is prong-split. The reduced shadow-unfold workflow currently compares that against an inclusive fake estimate.

The latest run shows:

| Configuration | Variable | Inclusive fakes | 1-prong fakes | 3-prong fakes | Prong-sum fakes |
|---|---|---:|---:|---:|---:|
| no_shadow_bin | MTW | 153.018 | 179.428 | 2.431 | 181.859 |
| MTW_shadow_bin_200 | MTW | 247.637 | 238.069 | 17.925 | 255.994 |
| MTW_shadow_bin_250 | MTW | 233.220 | 223.497 | 17.405 | 240.902 |
| MTW_shadow_bin_300 | MTW | 219.925 | 216.693 | 13.740 | 230.433 |
| TauPt_shadow_bin_250 | TauPt | 152.952 | 179.433 | 2.436 | 181.869 |

For the `MTW` shadow-bin cases, inclusive and prong-split fakes are similar. Switching to prong-split makes the subtraction slightly larger, not smaller. For no-shadow and `TauPt`, the prong-split sum is noticeably larger than the inclusive estimate, so it pushes the data-derived signal even lower.

After this run, `analysis_shadow_unfold.py` was updated so the nominal fake estimate is the prong-split method. The inclusive estimate is now only a diagnostic cross-check.

Representative prong-split comparison:

<img src="outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/fake_diagnostics/MTW_shadow_bin_250_MTW_inclusive_vs_prong_split_fakes.png" width="520">

### Interpretation

The closure picture is now split into two separate questions:

1. **Does the response unfold signal MC back to truth?**  
   Yes. The same-sample closure is exact by construction, and the split-sample closure is reasonable.

2. **Does the data-minus-background input have the same normalisation as signal MC?**  
   Not with the current full fake subtraction. The fake estimate is large enough to make the unfolded data sit below the signal prediction.

This means the remaining issue is not mainly a shadow-bin response issue. The next useful validation target is the fake estimate itself, especially the data control-region composition and MC-background subtraction used to form the fake factors.

## Dedicated Fake-Validation Script

A separate validation script was added and run:

`run/2017/validate_shadow_fakes.py`

Output summary:

`outputs/validate_shadow_fakes/shadow_fake_validation_summary.md`

Latest log:

`outputs/validate_shadow_fakes/logs/validate_shadow_fakes_2026-06-21_20-24-01.log`

The script reused the measured histogram cache from:

`outputs/analysis_shadow_unfold/measured`

The first validation tables reuse existing histogram caches and are cheap follow-up checks. The later staged `wtaunu_had` prong-balance check adds one targeted `wtaunu_had` dataframe pass, because the truth-fiducial and reco-preselection prong-split yields were not part of the cached unfolding outputs.

### Scope

The validation run is intentionally narrower than the unfolding script:

- variable under test: `MTW`;
- fake-factor source variable: `TauPt`;
- tau ID: `medium`;
- prongs: separate 1-prong and 3-prong fake estimates;
- configurations checked: `no_shadow_bin`, `MTW_shadow_bin_200`, `MTW_shadow_bin_300`.

The `MTW_shadow_bin_250` configuration was not included in this quick validation run. The 200 and 300 GeV shadow cases are enough to test whether the behaviour is generic across relaxed MTW thresholds.

### Control-region composition

The fake-factor control regions are fake-dominated after MC-contamination subtraction. For example:

| Configuration | Prong | Region | Data | MC contamination | Jet-fake-like yield | MC contamination / data |
|---|---|---|---:|---:|---:|---:|
| no_shadow_bin | 1-prong | CR_passID | 782.000 | 105.464 | 676.536 | 0.135 |
| no_shadow_bin | 1-prong | CR_failID | 13233.000 | 198.690 | 13034.309 | 0.015 |
| no_shadow_bin | 3-prong | CR_passID | 167.000 | 42.844 | 124.156 | 0.257 |
| no_shadow_bin | 3-prong | CR_failID | 10339.000 | 148.351 | 10190.649 | 0.014 |
| MTW_shadow_bin_300 | 1-prong | CR_passID | 2827.000 | 371.587 | 2455.413 | 0.131 |
| MTW_shadow_bin_300 | 1-prong | CR_failID | 37919.000 | 547.975 | 37371.025 | 0.014 |
| MTW_shadow_bin_300 | 3-prong | CR_passID | 510.000 | 132.642 | 377.358 | 0.260 |
| MTW_shadow_bin_300 | 3-prong | CR_failID | 21304.000 | 326.524 | 20977.476 | 0.015 |

This suggests the fake-factor construction regions are not dominated by simulated real-object contamination. The anti-ID denominators are especially clean by this metric.

Representative control-region plots:

| 1-prong CR pass-ID | 3-prong CR pass-ID |
|---|---|
| <img src="outputs/validate_shadow_fakes/plots/MTW_shadow_bin_300/1-prong/composition/MTW_shadow_bin_300_1-prong_CR_passID_TauPt_composition.png" width="390"> | <img src="outputs/validate_shadow_fakes/plots/MTW_shadow_bin_300/3-prong/composition/MTW_shadow_bin_300_3-prong_CR_passID_TauPt_composition.png" width="390"> |

### Fake-factor shapes

The 1-prong fake factor is consistently larger than the 3-prong fake factor over most of `TauPt`:

<img src="outputs/validate_shadow_fakes/plots/MTW_shadow_bin_300/fake_factors/MTW_shadow_bin_300_TauPt_prong_fake_factors.png" width="520">

That is not inherently wrong. It means the prong split matters numerically, and it is better to keep the prong-split method rather than collapse the fake factor into one inclusive estimate.

### Pass-ID fake validation

The decisive check is the pass-ID target:

```text
target = data - MC-contamination
```

in the pass-ID signal selection. This is compared directly to the prong-split fake prediction:

| Configuration | Variable | Data - MC-contamination | Prong-split fakes | Prediction / target |
|---|---|---:|---:|---:|
| no_shadow_bin | MTW | 104.812 | 181.859 | 1.735 |
| MTW_shadow_bin_200 | MTW | 101.604 | 255.994 | 2.520 |
| MTW_shadow_bin_300 | MTW | 107.336 | 230.433 | 2.147 |

The fake prediction is therefore too large compared with the pass-ID fake-like residual in data. This is the direct counterpart of the low unfolded-data normalisation seen in `analysis_shadow_unfold.py`.

Representative pass-ID validation plots:

| No shadow bin | MTW shadow 200 | MTW shadow 300 |
|---|---|---|
| <img src="outputs/validate_shadow_fakes/plots/no_shadow_bin/MTW/pass_id/no_shadow_bin_MTW_pass_id_fake_validation.png" width="300"> | <img src="outputs/validate_shadow_fakes/plots/MTW_shadow_bin_200/MTW/pass_id/MTW_shadow_bin_200_MTW_pass_id_fake_validation.png" width="300"> | <img src="outputs/validate_shadow_fakes/plots/MTW_shadow_bin_300/MTW/pass_id/MTW_shadow_bin_300_MTW_pass_id_fake_validation.png" width="300"> |

The shape agreement around the populated peak is not disastrous, but the fake prediction has too much total yield, especially in the relaxed MTW shadow configurations. The high prediction/target ratio is large enough to explain why applying the full fake subtraction drives the unfolded data below the signal-MC expectation.

### Fake-source variable cross-check

The validation script was then extended to rebuild the same pass-ID `MTW` fake prediction using `MTW` itself as the fake-factor source variable, instead of the nominal `TauPt` source. This is not yet a replacement prescription. It is a diagnostic of whether the `TauPt`-only fake-factor parameterisation is driving the over-subtraction.

| Configuration | Target variable | `TauPt`-sourced fakes | `MTW`-sourced fakes | `MTW / TauPt` |
|---|---|---:|---:|---:|
| no shadow bin | `MTW` | 181.859 | 100.966 | 0.555 |
| MTW shadow bin 200 | `MTW` | 255.994 | 118.256 | 0.462 |
| MTW shadow bin 300 | `MTW` | 230.433 | 109.590 | 0.476 |

This is an important result. The pass-ID `data - MC-contamination` targets are about `105`, `102`, and `107` events for the three rows above. The `MTW`-sourced fake estimates are therefore much closer to the independent pass-ID residual than the nominal `TauPt`-sourced estimates. The current evidence points to fake-factor transfer, not the event-weight calculation, as the next analysis-level issue to understand.

This does not mean the analysis should simply switch to an `MTW` fake factor. Since `MTW` is the measured observable, using it as the fake-factor source can sculpt the final spectrum if not justified carefully. The defensible next step is to test transfer stability in independent sidebands and, if needed, introduce a two-dimensional fake factor such as `(TauPt, MTW)` or `(TauPt, MET_met)` rather than choosing one source variable by eye.

### Prong-split fake-factor stability audit

A follow-up audit then checked whether the prong-split fake estimate itself is pathological. This did not rerun the ROOT event loops. It read the saved fake-factor internals from:

`outputs/analysis_shadow_unfold/measured/root/analysis_shadow_unfold_measured.root`

The check inspected, separately for 1-prong and 3-prong candidates:

- the fake-like CR pass-ID numerator, `CR pass-ID data - CR pass-ID MC-contamination`;
- the fake-like CR fail-ID denominator, `CR fail-ID data - CR fail-ID MC-contamination`;
- the fake factor;
- the fake-like SR fail-ID input, `SR fail-ID data - SR fail-ID MC-contamination`;
- bins with negative numerators, negative denominators, tiny denominators, negative fake factors, or negative SR fail-ID inputs.

The integrated result is:

| Configuration | Prong | CR pass fake-like | CR fail fake-like | SR fail fake-like | Predicted fakes | Fake-factor range | Flagged denominator bins | Flagged numerator bins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no shadow bin | 1 | 676.536 | 13034.309 | 2368.935 | 179.433 | 0.02584 to 0.13511 | 1 | 0 |
| no shadow bin | 3 | 124.156 | 10190.649 | 709.811 | 2.436 | -0.13151 to 0.04416 | 1 | 2 |
| MTW shadow bin 200 | 1 | 38228.102 | 381417.538 | 2526.548 | 238.176 | 0.03081 to 0.13375 | 1 | 0 |
| MTW shadow bin 200 | 3 | 4448.128 | 105504.269 | 735.896 | 17.929 | 0.00763 to 0.06932 | 0 | 0 |
| MTW shadow bin 250 | 1 | 9913.252 | 120798.132 | 2506.601 | 223.590 | 0.03052 to 0.12295 | 0 | 0 |
| MTW shadow bin 250 | 3 | 1327.323 | 45502.313 | 730.947 | 17.409 | 0.00835 to 0.05821 | 0 | 0 |
| MTW shadow bin 300 | 1 | 2455.413 | 37371.025 | 2480.699 | 216.782 | 0.02668 to 0.12065 | 0 | 0 |
| MTW shadow bin 300 | 3 | 377.358 | 20977.476 | 728.334 | 13.745 | 0.00797 to 0.05674 | 0 | 0 |

The detailed flagged bins are:

| Configuration | Prong | `TauPt` bin | CR pass fake-like numerator | CR fail fake-like denominator | Fake factor | SR fail fake-like | Predicted fakes | Flags |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| no shadow bin | 1 | 170-200 | 2.032 | 47.457 | 0.04282 | 646.732 | 27.692 | denominator below 1% of prong total |
| no shadow bin | 3 | 170-200 | -1.389 | 10.559 | -0.13151 | 34.718 | -4.566 | denominator below 1% of prong total; numerator negative; fake factor negative |
| no shadow bin | 3 | 200-250 | -5.078 | 121.959 | -0.04164 | 66.145 | -2.754 | numerator negative; fake factor negative |
| MTW shadow bin 200 | 1 | 600-1000 | 70.256 | 2280.373 | 0.03081 | 83.711 | 2.579 | denominator below 1% of prong total |

This narrows the problem. The prong-split fake estimate is not generally unstable because of tiny or negative fail-ID denominators. The MTW-shadow configurations have positive 3-prong fake-factor numerators and denominators in all checked bins.

The genuinely pathological case is the no-shadow 3-prong fake factor in the two lowest `TauPt` bins. There, the CR pass-ID fake-like numerator becomes negative after subtracting MC-contamination. That produces negative fake factors and negative predicted fake contributions, which then cancel the positive high-`TauPt` 3-prong fake prediction. This explains why the no-shadow 3-prong fake contribution can look artificially tiny.

For the MTW-shadow workflow, however, the large fake subtraction is not caused by this no-shadow 3-prong pathology. The MTW-shadow fake estimate is instead dominated by positive, low-`TauPt`, 1-prong anti-ID events. The 3-prong part remains small and positive.

### Prong-balance and low-TauPt checks

The validation script was extended with three additional checks:

1. a pass-ID prong-balance check, comparing 3-prong/1-prong ratios for data, `wtaunu_had`, total MC-contamination, the fake-like validation target, and the fake prediction;
2. a low-`TauPt` dominance check, measuring how much of the fake prediction comes from `170 <= TauPt < 250 GeV`;
3. a staged `wtaunu_had` prong-balance check at truth fiducial, reco-preselection, and medium pass-ID stages.

The prong-balance check shows that `wtaunu_had` is substantially more 3-prong-heavy than data in the pass-ID signal region:

| Configuration | Component | 1-prong | 3-prong | 3-prong / 1-prong |
|---|---|---:|---:|---:|
| no shadow bin | data | 1056.000 | 295.000 | 0.279 |
| no shadow bin | `wtaunu_had` MC-contamination | 710.998 | 289.654 | 0.407 |
| no shadow bin | total MC-contamination | 892.630 | 353.482 | 0.396 |
| no shadow bin | data - MC-contamination target | 163.370 | -58.482 | -0.358 |
| no shadow bin | fake prediction | 179.433 | 2.436 | 0.014 |
| MTW shadow bin 200 | data | 1133.000 | 310.000 | 0.274 |
| MTW shadow bin 200 | `wtaunu_had` MC-contamination | 758.752 | 310.906 | 0.410 |
| MTW shadow bin 200 | total MC-contamination | 960.926 | 380.380 | 0.396 |
| MTW shadow bin 200 | data - MC-contamination target | 172.074 | -70.380 | -0.409 |
| MTW shadow bin 200 | fake prediction | 238.176 | 17.929 | 0.075 |
| MTW shadow bin 300 | data | 1108.000 | 307.000 | 0.277 |
| MTW shadow bin 300 | `wtaunu_had` MC-contamination | 746.959 | 301.906 | 0.404 |
| MTW shadow bin 300 | total MC-contamination | 939.432 | 368.156 | 0.392 |
| MTW shadow bin 300 | data - MC-contamination target | 168.568 | -61.156 | -0.363 |
| MTW shadow bin 300 | fake prediction | 216.782 | 13.745 | 0.063 |

This is the clearest current diagnostic. The pass-ID data have a 3-prong/1-prong ratio of about `0.27-0.28`, while `wtaunu_had` has a ratio of about `0.40-0.41`. Since `wtaunu_had` is the dominant MC-contamination component, this makes the total MC-contamination too 3-prong-heavy and drives the 3-prong fake-like target negative.

The low-`TauPt` check shows that the fake prediction is dominated by the first two `TauPt` bins, especially in 1-prong:

| Configuration | Prong | Predicted fakes, 170-250 GeV | Total predicted fakes | Fraction from 170-250 GeV |
|---|---|---:|---:|---:|
| no shadow bin | 1-prong | 120.757 | 179.433 | 0.673 |
| no shadow bin | 3-prong | -7.320 | 2.436 | -3.005 |
| MTW shadow bin 200 | 1-prong | 175.086 | 238.176 | 0.735 |
| MTW shadow bin 200 | 3-prong | 7.674 | 17.929 | 0.428 |
| MTW shadow bin 300 | 1-prong | 155.771 | 216.782 | 0.719 |
| MTW shadow bin 300 | 3-prong | 3.327 | 13.745 | 0.242 |

This means the large fake subtraction is not primarily a high-mass tail artefact. It is mostly produced by low-`TauPt` anti-ID events, transferred into the pass-ID signal prediction through the `TauPt` fake factor.

Together, these checks suggest two separate effects:

- the fake estimate is numerically dominated by low-`TauPt`, 1-prong anti-ID events;
- the pass-ID validation target is being made too small by a `wtaunu_had` MC-contamination prediction that is too 3-prong-heavy compared with data.

The staged `wtaunu_had` prong-balance check shows where that second effect comes from:

| Configuration | Stage | 1-prong | 3-prong | 3-prong / 1-prong |
|---|---|---:|---:|---:|
| no shadow bin | truth fiducial | 2311.126 | 1343.171 | 0.581 |
| no shadow bin | reco preselection | 1135.196 | 500.141 | 0.441 |
| no shadow bin | medium pass-ID | 711.104 | 289.725 | 0.407 |
| MTW shadow bin 200 | truth fiducial | 2368.002 | 1381.717 | 0.583 |
| MTW shadow bin 200 | reco preselection | 1215.014 | 530.071 | 0.436 |
| MTW shadow bin 200 | medium pass-ID | 758.870 | 310.977 | 0.410 |
| MTW shadow bin 300 | truth fiducial | 2351.254 | 1370.661 | 0.583 |
| MTW shadow bin 300 | reco preselection | 1198.294 | 517.611 | 0.432 |
| MTW shadow bin 300 | medium pass-ID | 747.066 | 301.977 | 0.404 |

This rules out a simple story where the 3-prong excess is introduced only by the medium tau-ID requirement. The `wtaunu_had` sample is already much more 3-prong-heavy at truth fiducial level, with a 3-prong/1-prong ratio of about `0.58`. Reco preselection and medium ID reduce that to about `0.40-0.41`, but this is still well above the data pass-ID ratio of about `0.27-0.28`.

The immediate implication is that the negative 3-prong jet-fake-like residual is not just a fake-factor-transfer problem. It is also a signal/MC-contamination modelling or normalisation problem in the ID-region validation target: the `wtaunu_had` subtraction is too large in 3-prong relative to data.

### Is the `wtaunu_had` 3-prong excess caused by fakes?

No, not at the truth-fiducial stage. The staged check uses the `wtaunu_had` sample, which is split in `run/2017/samples.py` with a `TruthTau_isHadronic` hard cut. The truth-fiducial rows then require `passTruth`, `TruthTau_isHadronic`, `TruthTau_nChargedTracks == 1 || 3`, and the truth-level fiducial cuts. No reconstructed tau candidate or fake-factor region is involved in that first row. That means the high truth-fiducial `3p/1p` ratio is not a jet-to-tau fake effect.

The observed truth-fiducial ratio is also high compared with the usual hadronic tau decay composition. The ATLAS hadronic-tau identification and calibration paper states that hadronic tau decays contain one or three charged pions in about 72% and 22% of cases, respectively, with the remainder mostly involving charged kaons. That corresponds to a rough inclusive hadronic `3p/1p` ratio of about `0.31`, before any analysis-specific kinematic sculpting. Our truth-fiducial ratio is about `0.58`, so the analysis selection is either strongly sculpting the charged-prong composition or there is a sample/weight/truth-definition issue to understand. Source: ATLAS Collaboration, *Identification and energy calibration of hadronically decaying tau leptons with the ATLAS experiment in pp collisions at sqrt(s)=8 TeV*, arXiv:1412.7086, lines describing the hadronic decay composition: https://arxiv.org/abs/1412.7086.

A second useful cross-check is the direct topological tau branching-fraction measurement from LEP. L3 measured the inclusive tau branching fractions into one-, three-, and five-prong final states as 85.274%, 14.556%, and 0.170%, respectively. This includes leptonic tau decays in the one-prong category, so it is not the exact hadronic-only comparison for this analysis, but it reinforces the same point: an unsculpted tau sample is not expected to be intrinsically 3-prong-heavy. Source: L3 Collaboration, *Measurement of the Topological Branching Fractions of the tau lepton at LEP*, arXiv:hep-ex/0107055: https://arxiv.org/abs/hep-ex/0107055.

ATLAS fake-factor work points to a separate issue. The Universal Fake Factor paper emphasises that jet-to-tau fake rates depend on the tau candidate transverse momentum and charged-particle decay multiplicity, and on the composition of fake sources such as light-quark, gluon, heavy-flavour, and pile-up jets. This supports keeping the fake estimate prong-split, but it does not explain why the truth-level `wtaunu_had` sample is already 3-prong-heavy. Source: ATLAS Collaboration, *Estimation of backgrounds from jets misidentified as tau-leptons using the Universal Fake Factor method with the ATLAS detector*, arXiv:2502.04156: https://arxiv.org/abs/2502.04156.

The high-mass `tau + MET` ATLAS resonance search is the closest analysis topology to this work: hadronic tau, missing transverse momentum, and transverse mass as the key observable. It confirms that this is the right class of comparison for fake and tau-modelling checks, more so than electron/muon Drell-Yan analyses. Source: ATLAS Collaboration, *Search for high-mass resonances in final states with a tau-lepton and missing transverse momentum with the ATLAS detector*, arXiv:2402.16576: https://arxiv.org/abs/2402.16576.

The practical conclusion is that two effects should now be separated:

1. The fake estimate is likely too large when applied to the pass-ID SR target.
2. The `wtaunu_had` MC-contamination subtraction is itself too 3-prong-heavy relative to data.

The second effect is not a fake-background problem. It is a signal/MC-contamination modelling or fiducial-definition problem, and it must be understood before applying any empirical fake rescaling.

The follow-up truth-cut diagnostic has now been run in `validate_shadow_fakes.py`. It measures the `wtaunu_had` `3p/1p` ratio before fiducial cuts, then adds the truth cuts one at a time. It does this with weighted yields and raw event counts, and splits the result into `full` and `lm_cut` subsamples.

For the nominal no-shadow selection:

| Truth stage | All weighted 3p/1p | All unweighted 3p/1p | `full` weighted 3p/1p | `lm_cut` weighted 3p/1p |
|---|---:|---:|---:|---:|
| `passTruth` + hadronic prong | 0.306 | 0.306 | 0.307 | 0.306 |
| + `VisTruthTauPt` | 0.624 | 0.549 | 0.604 | 0.626 |
| + `TruthMTW` | 0.588 | 0.512 | 0.585 | 1.343 |
| + `TruthNeutrinoPt` | 0.583 | 0.510 | 0.583 | 0.454 |
| + truth eta | 0.581 | 0.508 | 0.581 | 0.444 |

The result is quite diagnostic. Before fiducial cuts, the weighted and unweighted `3p/1p` ratio is about `0.306`, which is consistent with the rough hadronic tau expectation from the ATLAS tau performance paper. The large ratio appears after the visible truth tau `pT` requirement. This is also visible in raw event counts, so it is not just a cross-section or event-weight artefact.

The `lm_cut` subsample behaves strangely after the `TruthMTW` cut, but its absolute final yield is tiny compared with `full` after the full fiducial selection. For example, after all nominal truth cuts, `lm_cut` contributes only about `3.218` weighted 1-prong and `1.429` weighted 3-prong events, while `full` contributes about `2307.908` and `1341.742`. The inclusive fiducial prong balance is therefore controlled by the `full` subsample.

This changes the interpretation. The high truth-fiducial `wtaunu_had` `3p/1p` ratio is mainly a hard visible-tau-`pT` acceptance effect, not a fake-background problem and not primarily a generator-weight pathology. However, this does not fully absolve the signal modelling: after reconstruction and medium tau ID, `wtaunu_had` still has an ID-region `3p/1p` ratio of about `0.40-0.41`, while data are about `0.27-0.28`. The remaining issue is therefore most likely in the reconstructed signal/MC-contamination modelling chain: tau-prong reconstruction, prong-dependent tau-ID efficiency, scale factors, or the selected signal kinematics.

The validation script was then extended again to compare weighted and unweighted reconstructed `wtaunu_had` prong ratios. This directly tests whether the reconstructed 3-prong excess is already present in raw event counts, or whether it is amplified by `reco_weight`.

| Configuration | Stage | Weighted 3p/1p | Unweighted 3p/1p | Weighted/unweighted shift |
|---|---|---:|---:|---:|
| no shadow bin | reco preselection, truth matched | 0.441 | 0.333 | 1.324 |
| no shadow bin | medium pass-ID, truth matched | 0.407 | 0.318 | 1.281 |
| MTW shadow bin 200 | reco preselection, truth matched | 0.436 | 0.332 | 1.315 |
| MTW shadow bin 200 | medium pass-ID, truth matched | 0.410 | 0.317 | 1.292 |
| MTW shadow bin 300 | reco preselection, truth matched | 0.432 | 0.332 | 1.303 |
| MTW shadow bin 300 | medium pass-ID, truth matched | 0.404 | 0.317 | 1.273 |

Truth matching itself has almost no effect: the all-reco and truth-matched rows are nearly identical in the validation output. The larger effect is the event weighting. In raw reconstructed counts, the medium pass-ID `wtaunu_had` ratio is about `0.317-0.318`; after `reco_weight`, it becomes about `0.404-0.410`. That is a `27-29%` increase in the 3-prong/1-prong ratio. This does not prove that tau scale factors alone are responsible, because `reco_weight` is the full reconstructed MC weight used by the framework. But it does show that the residual 3-prong excess is materially amplified by the reconstructed weighting chain rather than by truth matching.

The validation script now also splits the same reconstructed selections by weight definition. For the nominal no-shadow, medium pass-ID, truth-matched `wtaunu_had` selection:

| Weight definition | 3p/1p | Ratio / raw ratio |
|---|---:|---:|
| raw count | 0.318 | 1.000 |
| `mcWeight` | 0.297 | 0.934 |
| DTA `weight` branch | 0.333 | 1.045 |
| `truth_weight` | 0.355 | 1.117 |
| `reco_weight` | 0.407 | 1.281 |

The validation script was then extended one more time to break this down by DSID. That changes the interpretation. The aggregate table above makes it look as if the jump appears only when the framework applies the DSID/luminosity factor, but the per-DSID table shows the dominant effect is already present in the ntuple `weight` branch for the dominant high-mass sample. The per-DSID luminosity factor changes the relative importance of each DSID, but it does not change the `3p/1p` ratio within a DSID.

In code, `DatasetBuilder` defines:

```text
truth_weight = mcWeight * lumi * dsid_pmgf[mcChannel] / dsid_sumw[mcChannel]
reco_weight  = weight    * lumi * dsid_pmgf[mcChannel] / dsid_sumw[mcChannel]
```

The useful diagnostic is therefore to compare `raw`, `mcWeight`, the DTA `weight` branch, and `reco_weight` for each DSID. The top contributors after medium pass-ID are:

| Configuration | DSID | Physics short | 1-prong reco-weighted | 3-prong reco-weighted | 3p/1p | Fraction of weighted 3-prong yield |
|---|---:|---|---:|---:|---:|---:|
| no shadow bin | 700451 | `Sh_2211_Wtaunu_mW_120_ECMS_CVetoBVeto` | 577.756 | 245.464 | 0.425 | 0.847 |
| no shadow bin | 700450 | `Sh_2211_Wtaunu_mW_120_ECMS_CFilterBVeto` | 83.255 | 29.450 | 0.354 | 0.102 |
| no shadow bin | 700449 | `Sh_2211_Wtaunu_mW_120_ECMS_BFilter` | 10.157 | 5.266 | 0.518 | 0.018 |
| no shadow bin | 700348 | `Sh_2211_Wtaunu_H_maxHTpTV2_CFilterBVeto` | 4.818 | 4.640 | 0.963 | 0.016 |
| MTW shadow bin 200 | 700451 | `Sh_2211_Wtaunu_mW_120_ECMS_CVetoBVeto` | 612.184 | 256.406 | 0.419 | 0.825 |
| MTW shadow bin 200 | 700450 | `Sh_2211_Wtaunu_mW_120_ECMS_CFilterBVeto` | 91.379 | 31.054 | 0.340 | 0.100 |
| MTW shadow bin 300 | 700451 | `Sh_2211_Wtaunu_mW_120_ECMS_CVetoBVeto` | 605.393 | 255.162 | 0.421 | 0.845 |
| MTW shadow bin 300 | 700450 | `Sh_2211_Wtaunu_mW_120_ECMS_CFilterBVeto` | 88.285 | 30.549 | 0.346 | 0.101 |

This is quite sharp. The weighted 3-prong MC-contamination subtraction is dominated by DSID `700451`, the `mW_120_ECMS_CVetoBVeto` inclusive high-mass sample. It contributes about `83-85%` of the weighted 3-prong yield after medium pass-ID. The high-HT `CFilterBVeto` DSID `700348` has a very large `3p/1p` ratio, but it contributes only about `1.6-1.9%` of the weighted 3-prong yield, so it is not the main driver.

This statement needs one important qualification: DSID `700451` dominates the weighted 3-prong yield mostly because it dominates the selected `wtaunu_had` event yield. The more useful diagnostic is therefore the within-DSID 3-prong fraction, alongside the fraction of the selected event yield supplied by each DSID. The combined selected-yield table is:

| Configuration | DSID | Physics short | Raw selected yield frac | Raw 3-prong fraction | Reco selected yield frac | Reco 3-prong fraction | Reco/raw 3-prong fraction shift |
|---|---:|---|---:|---:|---:|---:|---:|
| no shadow bin | 700451 | `Sh_2211_Wtaunu_mW_120_ECMS_CVetoBVeto` | 0.919 | 0.243 | 0.823 | 0.298 | 1.228 |
| no shadow bin | 700450 | `Sh_2211_Wtaunu_mW_120_ECMS_CFilterBVeto` | 0.069 | 0.226 | 0.113 | 0.261 | 1.158 |
| no shadow bin | 700349 | `Sh_2211_Wtaunu_H_maxHTpTV2_CVetoBVeto` | 0.004 | 0.220 | 0.036 | 0.119 | 0.542 |
| no shadow bin | 700449 | `Sh_2211_Wtaunu_mW_120_ECMS_BFilter` | 0.005 | 0.240 | 0.015 | 0.341 | 1.420 |
| no shadow bin | 700348 | `Sh_2211_Wtaunu_H_maxHTpTV2_CFilterBVeto` | 0.001 | 0.190 | 0.009 | 0.491 | 2.576 |
| MTW shadow bin 200 | 700451 | `Sh_2211_Wtaunu_mW_120_ECMS_CVetoBVeto` | 0.917 | 0.242 | 0.812 | 0.295 | 1.219 |
| MTW shadow bin 200 | 700450 | `Sh_2211_Wtaunu_mW_120_ECMS_CFilterBVeto` | 0.069 | 0.225 | 0.114 | 0.254 | 1.129 |
| MTW shadow bin 200 | 700349 | `Sh_2211_Wtaunu_H_maxHTpTV2_CVetoBVeto` | 0.005 | 0.230 | 0.037 | 0.132 | 0.575 |
| MTW shadow bin 200 | 700449 | `Sh_2211_Wtaunu_mW_120_ECMS_BFilter` | 0.005 | 0.237 | 0.022 | 0.497 | 2.093 |
| MTW shadow bin 200 | 700348 | `Sh_2211_Wtaunu_H_maxHTpTV2_CFilterBVeto` | 0.001 | 0.208 | 0.011 | 0.516 | 2.485 |
| MTW shadow bin 300 | 700451 | `Sh_2211_Wtaunu_mW_120_ECMS_CVetoBVeto` | 0.919 | 0.242 | 0.820 | 0.297 | 1.224 |
| MTW shadow bin 300 | 700450 | `Sh_2211_Wtaunu_mW_120_ECMS_CFilterBVeto` | 0.069 | 0.225 | 0.113 | 0.257 | 1.142 |
| MTW shadow bin 300 | 700349 | `Sh_2211_Wtaunu_H_maxHTpTV2_CVetoBVeto` | 0.004 | 0.218 | 0.036 | 0.126 | 0.579 |
| MTW shadow bin 300 | 700449 | `Sh_2211_Wtaunu_mW_120_ECMS_BFilter` | 0.005 | 0.238 | 0.016 | 0.311 | 1.307 |
| MTW shadow bin 300 | 700348 | `Sh_2211_Wtaunu_H_maxHTpTV2_CFilterBVeto` | 0.001 | 0.218 | 0.010 | 0.535 | 2.457 |

The corrected interpretation is therefore more nuanced. DSID `700451` is not intrinsically bizarre in raw prong composition: its raw selected 3-prong fraction is about `24.3%`, close to the neighbouring `mW_120` slices. It matters because it is by far the largest selected sample. The reconstructed event weighting then raises its internal 3-prong fraction to about `29.8%`. Smaller DSIDs such as `700348` and `700449` have larger internal weighting shifts, but they are too small to dominate the total. The global pronginess issue is therefore the combination of a moderate weighting-induced shift in the dominant DSID plus larger shifts in low-yield DSIDs.

For DSID `700451` in the nominal no-shadow, medium pass-ID, truth-matched selection:

| Weight definition | 3p/1p | Ratio / raw ratio |
|---|---:|---:|
| raw count | 0.321 | 1.000 |
| `mcWeight` | 0.372 | 1.161 |
| DTA `weight` branch | 0.425 | 1.325 |
| `truth_weight` | 0.372 | 1.161 |
| `reco_weight` | 0.425 | 1.325 |

The latest validation run then decomposed the exposed components of the DTA `weight` branch by dividing them out one at a time. This is a diagnostic only: it identifies which branch component changes the prong composition, but it is not an analysis prescription.

| Configuration | Scope | Weight expression | 3p/1p | 3-prong fraction |
|---|---|---|---:|---:|
| no shadow bin | all DSIDs | `weight` | 0.333 | 0.250 |
| no shadow bin | all DSIDs | `weight / TauRecoSF` | 0.333 | 0.250 |
| no shadow bin | all DSIDs | `weight / selectionSF` | 0.292 | 0.226 |
| no shadow bin | all DSIDs | `weight / TriggerSF` | 0.292 | 0.226 |
| no shadow bin | all DSIDs | `weight / prwWeight` | 0.341 | 0.254 |
| no shadow bin | all DSIDs | `weight / jvtSF` | 0.333 | 0.250 |
| no shadow bin | all DSIDs | `weight / fjvtSF` | 0.329 | 0.248 |
| no shadow bin | DSID 700451 | `weight` | 0.425 | 0.298 |
| no shadow bin | DSID 700451 | `weight / TauRecoSF` | 0.425 | 0.298 |
| no shadow bin | DSID 700451 | `weight / selectionSF` | 0.373 | 0.272 |
| no shadow bin | DSID 700451 | `weight / TriggerSF` | 0.373 | 0.272 |
| no shadow bin | DSID 700451 | `weight / prwWeight` | 0.423 | 0.297 |
| no shadow bin | DSID 700451 | `weight / jvtSF` | 0.426 | 0.299 |
| no shadow bin | DSID 700451 | `weight / fjvtSF` | 0.425 | 0.298 |

The same pattern holds in the MTW shadow-bin selections. Removing `TauRecoSF`, `prwWeight`, `jvtSF`, or `fjvtSF` has little effect on the selected prong balance. Removing either `selectionSF` or `TriggerSF` lowers the 3-prong fraction by about `9%` relative for DSID `700451`, from `0.298` to `0.272` in the no-shadow selection. In the current ntuples those two branches appear numerically identical for this diagnostic, so this check does not yet distinguish whether the effect should be attributed to the tau-selection scale factor, trigger scale factor, or an aliasing/branch-definition convention in the DTA output.

An additional read-only branch audit was then performed on the dominant DSID `700451` ntuples. In the checked `mW_120_ECMS_CVetoBVeto` files:

| Quantity checked | Result |
|---|---|
| `selectionSF - TriggerSF` | exactly zero over the checked entries |
| `selectionSF / TriggerSF` | exactly `1.0` |
| `TauRecoSF` in the selected subset | exactly `1.0` |
| `weight / (mcWeight * prwWeight * jvtSF * fjvtSF * selectionSF)` | exactly `1.0` |
| mean `selectionSF` for selected 1-prong candidates | `0.812` |
| mean `selectionSF` for selected 3-prong candidates | `0.934` |

This resolves one ambiguity. The nominal event weight does not appear to double-count both `selectionSF` and `TriggerSF`; they are numerically the same branch content in this sample, and the DTA `weight` branch contains that factor once. However, that factor is strongly prong-dependent in the selected phase space. It suppresses selected 1-prong events more than selected 3-prong events, increasing the weighted `3p/1p` ratio.

The thesis section "Corrections to Monte Carlo Simulation" describes trigger scale factors as data-driven corrections for mismodelling of trigger efficiency in simulation. It does not define a separate `selectionSF` branch. In the framework, `selectionSF` is also used as the nominal denominator when constructing tau-efficiency systematic weight variations:

```text
weight_TAUS_TRUEHADTAU_EFF_* -> weight_TAUS_TRUEHADTAU_EFF_* * reco_weight / selectionSF
```

That code-level treatment is consistent with `selectionSF` representing the nominal tau/selection efficiency factor being replaced by a systematic variation. The branch audit shows that, at least for the dominant 700451 sample, this nominal factor is identical to `TriggerSF` in the ntuple. The safest interpretation is therefore: the prong-dependent factor is real in the current ntuple weights, but the branch naming is not sufficiently transparent to tell whether it is purely trigger efficiency, a combined selection efficiency, or an alias produced upstream in the DTA making.

So the residual reconstructed prong excess is not mainly an `lm_cut` problem, not mainly a small high-HT-slice problem, and not caused by the framework luminosity factor alone. It is dominated by the main `mW_120_ECMS_CVetoBVeto` W→τν sample, where the DTA `weight` branch and the resulting `reco_weight` raise the medium pass-ID `3p/1p` ratio from about `0.32` to about `0.43`. Within the exposed weight components, the only large lever arm currently visible is the prong-dependent `selectionSF`/`TriggerSF` factor.

### Interpretation

The dedicated validation confirms the fake-subtraction concern:

1. The fake-factor control regions are fake-dominated, so the method is not obviously failing because the CR is prompt-dominated.
2. The prong-split fake prediction is nevertheless too large when tested against the pass-ID fake-like residual.
3. The overprediction is strongest in the MTW shadow-bin selections, where the fake estimate is about `2.1-2.5` times the pass-ID residual.
4. The 3-prong part of the pass-ID residual is negative because the MC-contamination subtraction, dominated by `wtaunu_had`, exceeds data.
5. The `wtaunu_had` 3-prong-heavy truth-fiducial composition is created mainly by the visible truth tau `pT` cut. The remaining reconstructed medium-ID excess is dominated by DSID `700451`, where the DTA `weight` branch raises the `3p/1p` ratio from about `0.32` raw to about `0.43` after `reco_weight`.
6. The exposed weight-component diagnostic points specifically to the combined `selectionSF`/`TriggerSF` part of the DTA `weight` branch; pile-up and JVT-style components are not the main cause.
7. A read-only branch audit shows that `selectionSF` and `TriggerSF` are numerically identical for the dominant 700451 sample, and that `weight` contains this factor once, not twice.
8. This provides an unfolding-independent reason for the unfolded data deficit.

The fake factor may still be too aggressive, but the validation target itself is already distorted by the 3-prong MC-contamination subtraction. The next check should no longer be a broad DSID scan. It should be a targeted scale-factor audit upstream of this repository: verify exactly how `selectionSF` and `TriggerSF` are defined in the DTA ntuple production, why they are identical here, and whether the observed prong-dependent correction is expected for the tau-trigger/selection configuration used in this analysis.

## Diagnosing The Large Fake Subtraction

The current evidence points to the fake subtraction as the dominant remaining source of the unfolded-data normalisation deficit. This should now be treated as a fake-estimate validation problem, not as an unfolding-response closure problem.

### Method used in this analysis

The code implements the same fake-factor construction described in the thesis:

1. define a pass-ID tau region and an anti-ID tau region;
2. derive a fake factor in a fake-enriched control region;
3. subtract MC contamination from both numerator and denominator;
4. apply the fake factor to anti-ID signal-region data after subtracting MC-contamination;
5. perform the estimate separately for 1-prong and 3-prong taus, then sum the two estimates.

In compact form:

```text
FF = (CR pass-ID data - CR pass-ID MC-contamination)
   / (CR fail-ID data - CR fail-ID MC-contamination)

SR pass-ID fakes = (SR fail-ID data - SR fail-ID MC-contamination) * FF
```

The code implementation is in `Analysis.do_fakes_estimate`, where the fake factor is built from the control-region pass/fail histograms and applied as an event weight in the signal-region fail-ID dataframe. The current shadow-unfolding workflow uses `TauPt` as the fake-factor source variable and runs the method separately for 1-prong and 3-prong taus before summing them.

### What the thesis says

The thesis describes the same anti-ID fake-factor method in Chapter 8. It explicitly states that:

- the anti-ID region is defined by failing the tau JetRNN working-point requirement while passing a loose lower bound;
- true tau, muon, and electron contamination is subtracted using MC;
- the fake estimate is performed separately for 1-prong and 3-prong tau candidates;
- `TauPt` is kept as the default fake-factor binning because the `MTW`-binned estimate becomes unstable at high `MTW`;
- a flat 10% fake-estimate uncertainty was used, motivated by the high-mass tau-neutrino resonance analysis, but local validation studies were not done.

The important point is that the current code and thesis now agree at the formula level. The current problem is therefore not that the script is using a different fake-factor equation. The problem is that the assumptions behind the equation are not validated in the current shadow-bin phase space.

### What the validation currently shows

The dedicated validation script compares the prong-split fake prediction against:

```text
data - MC-contamination
```

in the pass-ID signal selection. This is the quantity the fake estimate should describe if the transfer from anti-ID to pass-ID is working.

| Configuration | Data - MC-contamination | Prong-split fakes | Prediction / target |
|---|---:|---:|---:|
| no shadow bin | 104.812 | 181.859 | 1.735 |
| MTW shadow bin 200 | 101.604 | 255.994 | 2.520 |
| MTW shadow bin 300 | 107.336 | 230.433 | 2.147 |

This means the fake prediction is too large compared with the pass-ID fake-like residual. The effect is present even without a shadow bin, but it becomes worse in the relaxed MTW shadow selections.

There is also a sharper warning sign in the prong breakdown: for 3-prong pass-ID events, the MC-contamination prediction is already larger than data before any fake estimate is applied. That gives a negative fake-like residual in the 3-prong pass-ID signal selection:

| Configuration | Prong | Data | MC-contamination | Data - MC-contamination |
|---|---:|---:|---:|---:|
| no shadow bin | 3-prong | 295.000 | 353.482 | -58.482 |
| MTW shadow bin 200 | 3-prong | 310.000 | 380.380 | -70.380 |
| MTW shadow bin 300 | 3-prong | 307.000 | 368.156 | -61.156 |

This makes the validation target artificially small. It means the fake estimate may be overpredicting, but the MC-contamination subtraction is also part of the problem.

### Comparison with ATLAS practice

The broad method is consistent with ATLAS practice for hadronic-tau analyses. ATLAS fake-factor methods estimate jet-to-tau backgrounds by measuring transfer factors in data control regions and applying them to anti-ID events in the signal phase space.

The most directly relevant recent ATLAS reference is:

- ATLAS Collaboration, "Estimation of backgrounds from jets misidentified as tau-leptons using the Universal Fake Factor method with the ATLAS detector", arXiv:2502.04156, EPJC 85 (2025) 1441.  
  https://arxiv.org/abs/2502.04156

This paper states that jets misidentified as hadronic tau decays are not reliably modelled by MC, motivating data-driven fake-factor methods. It also identifies the exact weakness we are now seeing: fake factors depend on the composition of fake sources, such as light-quark, gluon, heavy-flavour, and pile-up jets. If the fake composition differs between the region where the fake factor is measured and the region where it is applied, a single fake factor can mispredict the signal-region fake yield.

That is directly relevant here. Our control regions are fake-dominated, so the method is not obviously failing because of prompt contamination in the fake-factor denominator. The more likely failure mode is that the fake composition or kinematics in the CR fail/pass samples do not transfer cleanly to the high-MTW pass-ID signal selection, especially once the MTW shadow region is included.

The thesis also cites:

- ATLAS Collaboration, "Search for high-mass resonances in final states with a tau-lepton and missing transverse momentum with the ATLAS detector", Phys. Rev. D 109 (2024) 112008, arXiv:2402.16576.  
  https://arxiv.org/abs/2402.16576

This is close in final state to the present analysis: hadronic tau plus missing transverse momentum, with `MTW` as the key observable. It is a better methodological comparison for fake estimates than electron/muon Drell-Yan measurements, because the latter do not have the same jet-to-hadronic-tau fake problem.

### Formula-level audit against the Universal Fake Factor paper

Question:
Are we performing the fake-factor calculation itself incorrectly, or is the failure more likely due to the fake-factor determination region and transfer assumptions?

Implementation:
- code: `src/analysis.py`, `Analysis.do_fakes_estimate`
- workflow: `run/2017/analysis_shadow_unfold.py`
- mode: code audit only; no new ROOT event loops

The implemented fake-factor algebra matches the simpler fake-factor method described in the ATLAS Universal Fake Factor paper:

```text
FF = (DR pass-ID data - DR pass-ID MC-contamination)
   / (DR fail-ID data - DR fail-ID MC-contamination)

SR pass-ID fakes = (SR fail-ID data - SR fail-ID MC-contamination) * FF
```

In the code, the fake factor is built from the fake-factor determination-region pass/fail histograms and then applied as an event weight to the fail-ID signal-region dataframe. The current shadow-unfolding workflow then performs this separately for 1-prong and 3-prong tau candidates and sums the two estimates.

The important terminology from the ATLAS paper is that the fail-ID signal-region side is the anti-ID subregion of the SR, while the fake factors are measured in dedicated determination regions. In this repository, the names `sr_fail` and `fake_cr_pass/fake_cr_fail` correspond to those two pieces. The naming is slightly different from the paper, but the structure is the same.

The audit therefore does **not** point to a formula error. The stronger clue from the ATLAS paper is fake composition. The paper states that fake tau candidates can originate from light-quark, gluon, heavy-flavour, and pile-up jets, and that the fake factor depends on this composition. It also states that an ideal determination region would have the same fake-source composition as the anti-ID signal region, which is difficult in practice. The UFF method was introduced specifically to reduce this dependence by combining fake factors from several fake-type-enriched samples.

Interpretation:
The present analysis is using the simpler analysis-specific fake-factor method, not the full UFF construction. That is defensible, but it makes the method sensitive to whether the fake-factor determination region has the same fake composition and kinematic behaviour as the anti-ID SR. The current validation results suggest that this is the likely failure mode:

- the fake-factor construction itself is formula-level consistent with ATLAS practice;
- the control/determination regions are not obviously prompt-dominated;
- the fake prediction nevertheless over-subtracts the unfolded data input;
- MC fake-only closure does not show the same large overprediction;
- the low-MET and MET-binned validations show that fake-factor transfer across `MET_met` is not stable, especially for 3-prong candidates.

Conclusion:
The next analysis change should not be a correction to the fake-factor equation. It should be a validation-driven change to the fake-factor model or its assigned uncertainty: either an explicitly validated low-MET fake-enriched determination region, a MET-dependent transfer uncertainty, or a more differential fake-factor parameterisation if it can be made stable. The report should describe the present calculation as the standard simple fake-factor method, and the current limitation as a determination-region-to-signal-region transfer/composition problem.

For comparison, a recent ATLAS charged-current Drell-Yan cross-section measurement in electron and muon channels is:

- ATLAS Collaboration, "Measurement of double-differential charged-current Drell-Yan cross-sections at high transverse masses in pp collisions at sqrt(s)=13 TeV with the ATLAS detector", arXiv:2502.21088.  
  https://arxiv.org/abs/2502.21088

This is relevant for the high-MTW charged-current Drell-Yan phase space, but it is not directly comparable for the fake estimate because it measures `W -> e nu` and `W -> mu nu`, not hadronic tau final states. The fake-background problem in the current analysis is therefore more naturally compared with ATLAS hadronic-tau fake-factor methods than with electron/muon Drell-Yan fake estimates.

### Literature mapping for fake-factor transfer

Question:
What do nearby ATLAS analyses do to protect the fake-factor estimate against the issues seen here: MET transfer, prong dependence, fake-source composition, MC-contamination subtraction, and high-pT extrapolation?

Implementation:
- local scratchpad: `fake_factor_literature_mapping_20260622.md`
- workflow: literature/code comparison only; no new ROOT event loops
- closest sources:
  - ATLAS high-mass `tau + MET` search, arXiv:2402.16576: https://arxiv.org/abs/2402.16576
  - ATLAS Universal Fake Factor paper, arXiv:2502.04156: https://arxiv.org/abs/2502.04156
  - ATLAS high-mass `tau tau` Drell-Yan measurement, arXiv:2503.19836: https://arxiv.org/abs/2503.19836
  - ATLAS high-`mT` charged-current Drell-Yan `e/mu` measurement, arXiv:2502.21088: https://arxiv.org/abs/2502.21088
  - internal technical note: `/home/keanu/Uni_Stuff_Queen_Mary/Reading list/High-mass resonances to taunu may 21 NOTE.pdf`

The closest public analysis is the ATLAS high-mass `tau + MET` resonance search. It uses the same broad topology as this measurement: hadronic tau, missing transverse momentum, and transverse mass. Its fake estimate is built around three regions:

- an SR-like anti-ID region where events pass the signal selection but fail tau ID;
- a low-MET multijet-enriched pass-ID region;
- a low-MET multijet-enriched fail-ID region.

The public paper defines the jet background as jets misidentified as tau candidates, estimated from data, while non-jet backgrounds are estimated from simulation. It measures transfer factors in tau-pT and prong bins and applies them to the anti-ID region after subtracting simulated non-jet contamination. The internal note gives the same structure more explicitly: the multijet fake-factor regions use `MET < 100 GeV`, remove the `tau pT / MET` balance requirement, pass/fail tau ID while still passing the VeryLoose ID, and subtract real-lepton/non-jet contamination from simulation.

That is very close to the current candidate `lowMET_CR` in `analysis_shadow_unfold.py`, but with an important distinction: in the ATLAS analysis, the low-MET region is not assumed to transfer exactly. The paper and note assign dedicated systematics for:

- varying the low-MET fake-factor region, including changing the lower MET threshold from 0 to 30/50/70 GeV and the upper threshold from 100 to 150 GeV;
- non-jet subtraction in the fake-factor regions;
- quark/gluon composition differences between the anti-ID application region and the fake-factor determination region;
- high-tau-pT fake-factor extrapolation;
- statistical uncertainties in the fake factors;
- high-`mT` fake-shape extrapolation in the search tail.

The quark/gluon treatment is particularly relevant. The high-mass `tau + MET` paper assigns a 3-13% uncertainty by reweighting the tau-candidate jet seed width so that the anti-ID region better matches the multijet fail-ID fake-factor region. The Universal Fake Factor paper identifies this as the central weakness of the simple fake-factor method: fake factors depend on fake-source composition, including light-quark, gluon, heavy-flavour, and pile-up fake sources. The high-mass `tau tau` Drell-Yan measurement goes further, using tau-object width fits to determine source fractions and applying source-weighted fake factors.

The charged-current high-`mT` Drell-Yan `e/mu` paper is useful mainly as a methodological cross-check, not as a direct tau-fake model. It also uses data-driven fake/nonprompt estimates, fake-enriched regions, real-background subtraction, and validation regions. But because it measures electron and muon final states, the hadronic-tau fake-source and prong issues are better constrained by the tau+MET, UFF, and tau-tau papers.

Implications for this analysis:

| Issue | Literature treatment | Application here |
| --- | --- | --- |
| Fake-factor equation | Simple pass/fail fake factors with MC-contamination subtraction | Keep the current equation; do not rewrite the algebra. |
| Anti-ID application vs fake-factor determination | SR-like anti-ID application region is separate from low-MET determination regions | Keep `sr_fail` conceptually separate from `fake_cr_pass/fail`; document this clearly in the thesis. |
| MET transfer | Low-MET determination region plus MET-window variations | Treat `lowMET_CR` as a candidate method and build an MET-window envelope, not a one-shot replacement. |
| Prong dependence | Fake factors measured separately for 1-prong and 3-prong candidates | Keep prong-split fake factors as the baseline; inclusive-prong tests remain diagnostic. |
| Fake-source composition | Tau jet seed width / tau-object width used to assess quark/gluon/source composition | Audit whether a tau seed-width or tau-width branch exists in the ntuples. If not, use MET-window and tauPt/MET variations as proxy composition systematics. |
| MC-contamination subtraction | Non-jet/real contamination subtracted and assigned uncertainty | The negative high-MET 3-prong `data - MC-contamination` target should be treated as a subtraction/modelling limitation, not as evidence for negative fakes. |
| High-pT fake factors | Constant high-pT transfer-factor treatment assigned an extrapolation uncertainty | Check the highest `TauPt` fake-factor bin and add a tail stability uncertainty if it matters. |
| High-MET validation | Alternative high-MET validation region with `tau pT / MET < 0.7` and high `mT` | Add a validation-only region with high `MET_met`, high `MTW`, and `TauPt / MET_met < 0.7`. |
| MTW fake shape | Search-specific high-`mT` fake-shape fit and extrapolation | Do not use an `MTW` fake factor as nominal in an `MTW` unfolding; keep it diagnostic to avoid sculpting the measured observable. |

Recommendation:
The literature does not support a hand-tuned 3-prong scale factor or a nominal `MTW` fake factor as the next analysis change. It supports a more conservative fake-model programme:

1. keep the current fake-factor formula;
2. use the low-MET fake-enriched region as the leading alternative determination region;
3. validate it in a high-MET imbalanced `TauPt/MET_met < 0.7` region;
4. produce an MET-window systematic envelope;
5. audit whether tau seed-width/source-composition information is available;
6. only after those checks decide whether to change the nominal fake model or retain the current model with a larger transfer uncertainty.

The thesis wording should frame the current limitation as a fake-factor **transfer/composition** problem, not a formula-level bug.

### Completed diagnostics and next steps

The main diagnostics have now been done in `validate_shadow_fakes.py`:

1. the pass-ID SR MC-contamination was broken down by dataset and prong;
2. the prong-split fake prediction was broken down by `TauPt` source bin;
3. the `wtaunu_had` prong balance was tracked through the truth fiducial cuts;
4. the reconstructed `wtaunu_had` prong balance was compared before and after `reco_weight`;
5. the reconstructed `wtaunu_had` prong balance was split by raw counts, `mcWeight`, DTA `weight`, `truth_weight`, and `reco_weight`;
6. the same reconstructed prong balance was broken down by DSID;
7. the exposed DTA weight components were divided out one at a time to identify which component changes the reconstructed prong balance;
8. the dominant DSID `700451` ntuple branches were audited read-only to check whether `selectionSF`, `TriggerSF`, and `weight` are internally consistent.
9. the fake estimate was rebuilt using `MTW` rather than `TauPt` as the fake-factor source variable.
10. the saved prong-split fake-factor numerator and denominator histograms were audited for negative or tiny bins without rerunning the ROOT event loops.
11. a fast fake-transfer validation was run in independent MET and MTW sidebands, using cached sideband histograms where possible;
12. a focused MET-binned transfer validation was run for `MTW_shadow_bin_300`, deriving and applying separate `TauPt` fake factors in two MET slices while validating across an independent MTW sideband.

The result is more specific than the earlier suspicion. The fake estimate is too large against the pass-ID residual, but that residual is also biased by the `wtaunu_had` MC-contamination subtraction. The high truth-fiducial `wtaunu_had` `3p/1p` ratio comes mainly from the visible truth tau `pT` cut. The remaining pass-ID discrepancy is dominated by the main `mW_120_ECMS_CVetoBVeto` sample, DSID `700451`, where the DTA `weight` branch raises the reconstructed medium-ID `3p/1p` ratio from about `0.32` raw to about `0.43` after `reco_weight`. The component-removal diagnostic narrows that effect to the exposed `selectionSF`/`TriggerSF` branch content; removing pile-up or JVT-style components does not materially reduce the prong imbalance. The branch audit shows that `selectionSF` and `TriggerSF` are identical in the checked 700451 files and that `weight` contains this factor once, so this is not a simple double-counting error in the framework. Since the weight calculation has been verified and the upstream DTA code is not available, the most actionable remaining path is the fake-factor transfer: the `MTW`-sourced diagnostic reduces the fake estimate by roughly a factor of two and brings it close to the pass-ID residual.

The prong-split fake-factor audit adds an important constraint on that interpretation. The no-shadow 3-prong fake factor is pathological in the low-`TauPt` bins because the CR pass-ID numerator is negative after MC-contamination subtraction. That explains the unusually tiny no-shadow 3-prong fake contribution. But the MTW-shadow 3-prong fake factors are positive and have healthy denominators, so the main MTW-shadow fake over-subtraction is not explained by denominator instability or by the no-shadow 3-prong cancellation.

### Fake-transfer validation

The independent transfer validation has now been run in `validate_shadow_fakes.py`. The run produced a new sideband cache:

`outputs/validate_shadow_fakes/sideband_transfer/root/validate_shadow_fakes_sideband_transfer.root`

The first run took about 37 minutes because it had to build the sideband histograms and then evaluate each transfer fake estimate. Future runs with the same selections should be much faster because the sideband histogram file now exists.

The most useful rows are:

| Configuration | Prong | Transfer test | Predicted fakes | Validation target | Prediction / target |
|---|---:|---|---:|---:|---:|
| no shadow | 1 | MET split | 165.074 | 237.232 | 0.696 |
| no shadow | 3 | MET split | 30.080 | 13.060 | 2.303 |
| no shadow | 1 | SR MET proxy | 141.205 | 139.685 | 1.011 |
| no shadow | 3 | SR MET proxy | 0.673 | -43.199 | -0.016 |
| MTW shadow 200 | 1 | MET split | 689.544 | 628.020 | 1.098 |
| MTW shadow 200 | 3 | MET split | 81.497 | 24.071 | 3.386 |
| MTW shadow 200 | 1 | SR MET proxy | 192.612 | 144.519 | 1.333 |
| MTW shadow 200 | 3 | SR MET proxy | 12.482 | -48.199 | -0.259 |
| MTW shadow 200 | 1 | MTW split | 710.982 | 677.359 | 1.050 |
| MTW shadow 200 | 3 | MTW split | 132.220 | 124.117 | 1.065 |
| MTW shadow 300 | 1 | MET split | 526.257 | 524.028 | 1.004 |
| MTW shadow 300 | 3 | MET split | 61.417 | 25.819 | 2.379 |
| MTW shadow 300 | 1 | SR MET proxy | 175.237 | 139.946 | 1.252 |
| MTW shadow 300 | 3 | SR MET proxy | 9.277 | -45.169 | -0.205 |
| MTW shadow 300 | 1 | MTW split | 699.203 | 677.359 | 1.032 |
| MTW shadow 300 | 3 | MTW split | 128.217 | 124.117 | 1.033 |

The interpretation is more nuanced than "the fake factor is simply bad":

- The MTW-split transfer checks are very good, with prediction/target ratios around `1.03-1.07` for both prongs. This means the `TauPt` fake factor can transfer between the relaxed-MTW and nominal-MTW sidebands when the MET requirement remains in the control region.
- The 1-prong MET-split checks are acceptable in the MTW-shadow configurations, with ratios of `1.10` and `1.00`, but no-shadow 1-prong underpredicts the adjacent MET sideband.
- The 3-prong MET-split checks overpredict badly, with ratios of about `2.3-3.4`.
- The SR MET proxy still has negative 3-prong validation targets. That is not a fake-factor denominator problem; it is the same MC-contamination-subtraction problem seen in the pass-ID SR validation.

So the latest result does not justify a global fake rescaling. It points to a more specific issue: the `TauPt` fake factor transfers reasonably in MTW but poorly across MET, especially for 3-prong candidates. Since the final SR requires high MET, the MET dependence is likely relevant to the over-subtraction in the unfolded data input.

The no-shadow 3-prong negative-bin treatments were also tested from cached histograms:

| Treatment | Predicted fakes | Pass-ID target | Prediction / target |
|---|---:|---:|---:|
| nominal signed | 2.436 | -58.482 | -0.042 |
| floor negative bins | 9.756 | -58.482 | -0.167 |
| merge 170-250 GeV | 4.834 | -58.482 | -0.083 |

This confirms that low-`TauPt` negative-bin treatment cannot fix the main shadow-bin issue. It only changes the already-small no-shadow 3-prong prediction.

The next analysis development should therefore be a sideband-validated fake-factor parameterisation that includes MET dependence, for example `(TauPt, MET_met)` with prong splitting. A `(TauPt, MTW)` fake factor is still useful diagnostically, but it is less attractive as the nominal prescription because `MTW` is the measured observable. The main unfolding script should remain on the current prong-split `TauPt` method until this alternative is validated in `validate_shadow_fakes.py`.

### MET-binned fake-transfer validation

The next validation tested whether the observed MET-transfer failure can be improved by keeping the fake-factor source variable as `TauPt`, but deriving and applying separate fake factors in coarse MET slices. This was implemented in:

`run/2017/validations/validate_met_binned_transfer.py`

The validation uses only `MTW_shadow_bin_300`, medium tau ID, 1- and 3-prong taus, and `MTW` as the validation observable. It derives fake factors in the relaxed MTW sideband,

```text
300 <= MTW < 350
```

and validates them in the nominal MTW sideband,

```text
MTW >= 350
```

using the same MET slices on both sides. This makes the MET bins overlapping between derivation and validation, unlike the earlier independent `MET < 120` versus `120 <= MET < 170` test.

The first run produced:

`outputs/validate_shadow_fakes/met_binned_transfer/met_binned_transfer_summary.md`

and cached the validation histograms in:

`outputs/validate_shadow_fakes/met_binned_transfer/root/validate_met_binned_transfer.root`

The first run took about 7.5 minutes, from `15:58:11` to `16:05:44` in the log. A subsequent rerun loaded the existing dataset and fake-estimate caches and reported `event loops run in this invocation: False`, reproducing the same table below. Future reruns with the same selections should therefore be used as cache-only checks unless the validation selections or binning are deliberately changed.

| MET slice | Prong | CR numerator | CR denominator | Negative numerator bins | Tiny/non-positive denominator bins | Validation fail-ID input | Predicted fakes | Validation target | Prediction / target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MET < 120 | 1 | 1492.081 | 21517.951 | 0 | 1 | 9713.776 | 474.288 | 440.127 | 1.078 |
| MET < 120 | 3 | 240.443 | 10277.948 | 1 | 1 | 8511.522 | 93.905 | 111.056 | 0.846 |
| 120 <= MET < 170 | 1 | 286.796 | 2818.764 | 5 | 5 | 3320.533 | 124.403 | 237.232 | 0.524 |
| 120 <= MET < 170 | 3 | 12.758 | 508.880 | 5 | 5 | 1679.126 | -89.122 | 13.060 | -6.824 |

This is not the hoped-for clean rescue by MET binning.

The low-MET slice behaves reasonably: the 1-prong prediction is about `8%` high, and the 3-prong prediction is about `15%` low. That suggests the method can work in a fake-rich, low-MET sideband.

The `120 <= MET < 170` slice is the problem. The 1-prong prediction is only about half the validation target, and the 3-prong prediction is negative. The latter is not a harmless statistical fluctuation in the final integral: the source histograms already show five negative numerator bins and five tiny or non-positive denominator bins in this slice. This means the upper-MET control slice is too sparse or too contaminated after MC-contamination subtraction to support a stable prong-split `TauPt` fake factor in the current binning.

The conclusion changes slightly from the earlier "add MET dependence" hypothesis. The data still show that MET matters, but a naive MET-binned `TauPt` fake factor is not yet a defensible nominal correction. It improves the conceptual modelling axis but exposes a stability problem in the high-MET sideband.

The immediate next step should be a stability-first fake-model test, not a full unfolding rerun:

1. merge the upper MET control slice with looser or coarser `TauPt` bins, especially for 3-prong;
2. test a prong-dependent treatment where 1-prong may keep more binning, while 3-prong uses a coarser or inclusive fake factor in the upper MET slice;
3. compare this against a pure MET-threshold uncertainty, treating the difference between low-MET and upper-MET transfer as a systematic rather than a nominal correction;
4. only after a stable sideband result should the main `analysis_shadow_unfold.py` workflow be changed.

### MET cut choice validation

The next direct test asked whether the current fake-factor control region should simply be tightened from:

```text
MET < 170
```

to:

```text
MET < 120
```

This is a narrower question than the full MET-binned validation above. The comparison uses the same validation target for both fake-factor derivations:

```text
MTW >= 350
MET < 170
```

and compares fake factors derived in the same relaxed-MTW sideband,

```text
300 <= MTW < 350
```

with either `MET < 170` or `MET < 120`.

The output is:

`outputs/validate_shadow_fakes/met_cut_comparison/met_cut_comparison_summary.md`

and the thesis-ready comparison plots are:

| 1-prong | 3-prong |
|---|---|
| <img src="outputs/validate_shadow_fakes/met_cut_comparison/plots/met_cut_comparison/MTW_shadow_bin_300_1prong_MET_cut_fake_transfer_comparison.png" width="390"> | <img src="outputs/validate_shadow_fakes/met_cut_comparison/plots/met_cut_comparison/MTW_shadow_bin_300_3prong_MET_cut_fake_transfer_comparison.png" width="390"> |

The integral comparison is:

| Prong | Derivation region | CR numerator | CR denominator | Negative numerator bins | Tiny/non-positive denominator bins | Validation fail-ID input | Predicted fakes | Validation target | Prediction / target | `|ratio - 1|` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | derive `MET < 170` | 1778.877 | 24336.715 | 0 | 0 | 13034.309 | 699.203 | 677.359 | 1.032 | 0.032 |
| 1 | derive `MET < 120` | 1492.081 | 21517.951 | 0 | 1 | 13034.309 | 702.033 | 677.359 | 1.036 | 0.036 |
| 3 | derive `MET < 170` | 253.202 | 10786.827 | 1 | 0 | 10190.649 | 128.217 | 124.117 | 1.033 | 0.033 |
| 3 | derive `MET < 120` | 240.443 | 10277.948 | 1 | 1 | 10190.649 | 129.793 | 124.117 | 1.046 | 0.046 |

This test does **not** show an improvement from lowering the control-region MET cut. Both derivation choices predict the same validation target well, but `MET < 120` is slightly worse for both prongs by the integral metric:

- 1-prong: `MET < 170` gives `1.032`, while `MET < 120` gives `1.036`;
- 3-prong: `MET < 170` gives `1.033`, while `MET < 120` gives `1.046`.

The plots show the same result visually. The two predictions are almost on top of each other in the populated bins. The high-MTW tail remains statistically weak, but it is not where the control-region choice is being decided.

The important distinction is:

- deriving separate fake factors in a clean low-MET slice can behave sensibly;
- simply replacing the current `MET < 170` derivation with `MET < 120` does not materially improve this validation target.

So the evidence does not support a nominal analysis change to `MET < 120` on its own. If a lower-MET derivation is pursued, it should be treated as a more ATLAS-like transfer-factor strategy: derive in a cleaner fake-enriched region, validate the extrapolation to high MET, and assign a dedicated extrapolation uncertainty.

### Coarse 3-prong fake-factor validation

The next cache-only test asked whether the problematic 3-prong behaviour is mainly caused by the fine `TauPt` fake-factor binning. This was implemented in:

`run/2017/validations/validate_coarse_3prong_fakes.py`

It reads existing fake-factor numerator, denominator, and fail-ID input histograms and recomputes the 3-prong fake prediction after merging `TauPt` source bins. It does not rerun ROOT event loops and does not change the nominal fake model.

The output is:

`outputs/validate_shadow_fakes/coarse_3prong/coarse_3prong_fake_summary.md`

The tested source binnings were:

```text
nominal 8 bins: 170, 200, 250, 300, 350, 425, 500, 600, 1000
merge >=300:    170, 200, 250, 300, 1000
merge >=250:    170, 200, 250, 1000
two bins:       170, 250, 1000
inclusive:      170, 1000
```

The most relevant rows are:

| Test | Source binning | Predicted fakes | Validation target | Prediction / target | Negative numerator groups | Max `|FF|` |
|---|---|---:|---:|---:|---:|---:|
| independent MET split | nominal 8 bins | 61.423 | 25.819 | 2.379 | 0 | 0.0600 |
| independent MET split | merge >=250 | 56.535 | 25.819 | 2.190 | 0 | 0.0498 |
| independent MET split | inclusive | 40.935 | 25.819 | 1.585 | 0 | 0.0187 |
| upper-MET internal slice | nominal 8 bins | -89.122 | 13.060 | -6.824 | 5 | 0.4710 |
| upper-MET internal slice | merge >=250 | -0.750 | 13.060 | -0.057 | 1 | 0.0294 |
| upper-MET internal slice | inclusive | 42.098 | 13.060 | 3.223 | 0 | 0.0251 |
| derive `MET < 170` | nominal 8 bins | 128.217 | 124.117 | 1.033 | 1 | 0.0598 |
| derive `MET < 170` | merge >=250 | 220.969 | 124.117 | 1.780 | 0 | 0.0441 |
| derive `MET < 120` | nominal 8 bins | 129.793 | 124.117 | 1.046 | 1 | 0.0605 |
| derive `MET < 120` | merge >=250 | 222.749 | 124.117 | 1.795 | 0 | 0.0498 |

This does not support a coarser 3-prong fake factor as the next nominal model.

The coarser binning does remove some negative numerator groups in the pathological upper-MET slice, but it does not produce a reliable prediction. In the direct MET-cut comparison, where the nominal 3-prong estimate is already close to the validation target, coarsening makes the prediction substantially worse. For example, `derive MET < 170` moves from a ratio of `1.033` to `1.780` when the bins are merged at `250 GeV`.

The interpretation is:

- the upper-MET 3-prong instability is real;
- it is not fixed by simply merging the `TauPt` source bins;
- the nominal fine-binned 3-prong estimate is actually preferred in the direct MET-cut validation;
- a coarser 3-prong treatment could be useful as a systematic stress test, but not as a better central fake model.

The next useful fake-model test should therefore move away from pure 1D `TauPt` rebinning. The remaining plausible directions are:

1. an explicit MET-transfer uncertainty, using the difference between low-MET and upper-MET behaviour as a systematic rather than as a central correction;
2. an ATLAS-like fake-enriched low-MET derivation with a validated extrapolation uncertainty;
3. a true 2D or category-based fake-factor model only if it can be validated without sparse/negative bins.

### Inclusive-prong fake-factor validation

The next cache-only test asked whether combining 1-prong and 3-prong candidates into a single inclusive fake factor transfers better than the prong-split fake-factor method. This was implemented in:

`run/2017/validations/validate_inclusive_prong_fakes.py`

It reads saved 1-prong and 3-prong fake-factor numerator, denominator, and fail-ID input histograms, combines the prongs at the source-histogram level, and compares the resulting inclusive fake-factor prediction with the ordinary prong-split prediction. It does not run ROOT event loops.

The output is:

`outputs/validate_shadow_fakes/inclusive_prong/inclusive_prong_fake_summary.md`

The result is:

| Test | Model | Predicted fakes | Validation target | Prediction / target |
|---|---|---:|---:|---:|
| independent MET split | prong split | 587.575 | 549.847 | 1.069 |
| independent MET split | inclusive 1+3 | 587.720 | 549.847 | 1.069 |
| MTW split | prong split | 827.430 | 801.476 | 1.032 |
| MTW split | inclusive 1+3 | 821.917 | 801.476 | 1.026 |
| derive `MET < 170` | prong split | 827.430 | 801.476 | 1.032 |
| derive `MET < 170` | inclusive 1+3 | 821.917 | 801.476 | 1.026 |
| derive `MET < 120` | prong split | 831.836 | 801.476 | 1.038 |
| derive `MET < 120` | inclusive 1+3 | 826.431 | 801.476 | 1.031 |

This does not show evidence for switching to an inclusive 1+3 fake factor.

The inclusive fake factor is marginally closer to unity in the MTW split and MET-cut-choice checks, but the effect is small: a few events on an approximately 800-event validation target. In the independent MET split, inclusive and prong-split are essentially identical. The inclusive fake factor also does not address the earlier pass-ID over-subtraction problem; the improvement is too small compared with the factor-of-two-level discrepancy seen in the pass-ID fake validation.

The interpretation is:

- prong splitting is not the source of the main fake-transfer problem;
- collapsing to inclusive 1+3 does not provide a meaningful rescue;
- keeping prong splitting remains physically better motivated, unless a later systematic model needs an inclusive-prong stress test.

### Comparison with the ATLAS high-mass tau+MET fake control region

The closest published ATLAS comparison is the high-mass `tau + MET` resonance search:

- ATLAS Collaboration, "Search for high-mass resonances in final states with a tau-lepton and missing transverse momentum with the ATLAS detector", Phys. Rev. D 109 (2024) 112008, arXiv:2402.16576: https://arxiv.org/abs/2402.16576

That analysis also estimates the jet-to-tau background from data. Its control-region strategy is different from the one currently used here.

In the ATLAS search, the region where the fake estimate is applied, CR1, is signal-like but anti-ID:

```text
same selection as the signal region
tau fails loose ID but passes very-loose ID
```

The transfer factors themselves are not derived in that high-MET signal-like region. They are derived in two low-MET, dijet-enriched regions:

```text
CR2: tau passes loose ID
CR3: tau fails loose ID but passes very-loose ID
MET < 100 GeV
pT_tau / MET requirement removed
other selection criteria kept close to the signal selection
```

"Dijet-enriched" means the selected tau candidates are deliberately made more likely to be quark/gluon jets misidentified as taus. This is not just an accidental property of the phase space. ATLAS enriches the region by requiring low missing transverse momentum and removing the tau/MET balance requirement. Real `W -> tau nu`-like events tend to have genuine missing transverse momentum from neutrinos, while dijet and multijet events more often enter through jet mismeasurement and jet-to-tau misidentification.

The transfer factors are measured in `TauPt` intervals and separately for 1-prong and 3-prong tau candidates. ATLAS reports that additional dependences, including missing transverse momentum and trigger, were checked and found negligible for that analysis.

Our current fake-factor derivation is different:

```text
MTW > threshold
TauPt > 170
MET < 170
pass/fail medium tau ID
```

and it is applied to:

```text
MTW > threshold
TauPt > 170
MET > 170
fail medium tau ID
```

So our fake-factor control region is closer to the signal region in MET than the ATLAS transfer-factor derivation region. That sounds attractive, but the validation now shows that the upper part of this control region,

```text
120 <= MET < 170
```

is not stable after MC-contamination subtraction. In contrast, the lower-MET slice behaves more sensibly.

The ATLAS approach is therefore a useful conceptual guide. They do not derive the transfer factor as close as possible to the signal region at all costs. They derive it in a cleaner fake-enriched phase space, then validate the transfer and assign systematic uncertainties for the extrapolation to the signal region.

They explicitly assign systematic uncertainties for possible residual correlations between the transfer factors and the control-region definitions. These include threshold variations, non-jet subtraction uncertainty, quark/gluon composition differences, and extrapolation uncertainties for the high-`MTW` shape.

The corresponding lesson for this analysis is:

```text
Deriving in MET < 120 may be more stable than deriving in MET < 170,
but the extrapolation to MET > 170 must be validated and covered by a systematic.
```

That is a more defensible direction than forcing a separate `120 <= MET < 170` fake factor when the validation shows that region has negative/tiny source bins.

### Fake-factor size compared with the ATLAS high-mass tau+MET note

Question:
Are our fake rates comparable to the high-mass `tau + MET` fake-factor estimate?

The answer is: only qualitatively. Numerically, our current medium-ID fake factors are much smaller than the fake factors quoted in the ATLAS high-mass `tau + MET` note. This means the large fake subtraction seen in the current unfolding workflow is not caused by an unusually large transfer factor. It is more likely caused by the size and composition of the fail-ID application sample, or by how the MC-contamination subtraction is being applied.

The closest ATLAS note values are Table 14 of:

```text
/home/keanu/Uni_Stuff_Queen_Mary/Reading list/High-mass resonances to taunu may 21 NOTE.pdf
```

The public paper corresponding to that analysis is ATLAS Collaboration, "Search for high-mass resonances in final states with a tau-lepton and missing transverse momentum with the ATLAS detector", Phys. Rev. D 109 (2024) 112008, arXiv:2402.16576: https://arxiv.org/abs/2402.16576.

The ATLAS note quotes the following high-`pT` fake factors:

| Source | Prong | `TauPt` bin | Fake factor |
|---|---:|---:|---:|
| ATLAS high-mass note, Table 14 | 1 | 350-500 GeV | `0.362 +/- 0.016` |
| ATLAS high-mass note, Table 14 | 1 | 500-1000 GeV | `0.336 +/- 0.032` |
| ATLAS high-mass note, Table 14 | 3 | 350-500 GeV | `0.116 +/- 0.009` |
| ATLAS high-mass note, Table 14 | 3 | 500-1000 GeV | `0.083 +/- 0.015` |

The current `analysis_shadow_unfold.py` low-MET fake factors saved in:

```text
outputs/analysis_shadow_unfold/measured/root/analysis_shadow_unfold_measured.root
```

are:

| Source | Prong | `TauPt` bin | Fake factor |
|---|---:|---:|---:|
| this analysis, `no_shadow_bin_medium_1prong_lowMET_TauPt_FF` | 1 | 350-425 GeV | `0.055` |
| this analysis, `no_shadow_bin_medium_1prong_lowMET_TauPt_FF` | 1 | 425-500 GeV | `0.034` |
| this analysis, `no_shadow_bin_medium_1prong_lowMET_TauPt_FF` | 1 | 500-600 GeV | `0.034` |
| this analysis, `no_shadow_bin_medium_1prong_lowMET_TauPt_FF` | 1 | 600-1000 GeV | `0.027` |
| this analysis, `no_shadow_bin_medium_3prong_lowMET_TauPt_FF` | 3 | 350-425 GeV | `0.013` |
| this analysis, `no_shadow_bin_medium_3prong_lowMET_TauPt_FF` | 3 | 425-500 GeV | `0.0077` |
| this analysis, `no_shadow_bin_medium_3prong_lowMET_TauPt_FF` | 3 | 500-600 GeV | `0.0072` |
| this analysis, `no_shadow_bin_medium_3prong_lowMET_TauPt_FF` | 3 | 600-1000 GeV | `0.0084` |

So the current fake factors are lower than the note by roughly:

| Prong | Region | Approximate difference |
|---:|---|---:|
| 1 | 350-500 GeV | ATLAS note is about `6-11x` larger |
| 1 | 500-1000 GeV | ATLAS note is about `10-12x` larger |
| 3 | 350-500 GeV | ATLAS note is about `9-15x` larger |
| 3 | 500-1000 GeV | ATLAS note is about `10x` larger |

This is not necessarily a contradiction, because the two measurements are not defining the same conditional probability. Important differences are:

| Ingredient | ATLAS high-mass `tau + MET` note | Current shadow-unfolding workflow |
|---|---|---|
| Tau ID numerator | Loose ID | Medium ID |
| Anti-ID denominator | fail Loose, pass VeryLoose | fail Medium, `TauRNNJetScore > 0.01` |
| Application region | signal-like anti-ID region | high-MET fail-medium region used for unfolding subtraction |
| Derivation region | `MET < 100`, `pT_tau / MET` removed, multijet-enriched | low-MET candidate, but with this analysis's medium-ID definitions |
| Signal/search phase space | high-mass resonance search with balanced `0.7 < pT_tau / MET < 1.3` in SR | charged-current `W -> tau nu` unfolding with `MTW` as the measurement variable |
| Final treatment | fake estimate plus high-`mT` tail treatment and fake-transfer systematics | fake subtraction before fiducial unfolding |

The correction to the "they should be identical" intuition is therefore:

> They should show similar qualitative behaviour only if the fake-source composition is similar: prong-split, `TauPt` dependent, and generally decreasing at high `TauPt`. They should not be expected to have identical numerical values unless the tau-ID working point, anti-ID floor, control region, signal/application region, trigger/year mixture, and MC-contamination subtraction are also matched.

The current comparison still gives a useful diagnostic. Because our fake factors are much smaller than the ATLAS note values, the fake-subtraction tension is unlikely to be fixed by asking why our transfer factor is too large. The next direct check should be:

```text
fake yield = fake factor * (SR fail-ID data - SR fail-ID MC-contamination)
```

split by prong, `TauPt` source bin, and truth category. If the final fake yield is too large despite small fake factors, the fail-ID application input is the place to debug.

The most useful like-for-like validation would be to reproduce the note as closely as possible inside this framework:

1. build a validation-only Loose/VeryLoose fake-factor calculation;
2. use `MET < 100` and remove the `TauPt / MET` balance requirement;
3. bin exactly as Table 14, especially `350-500` and `500-1000` GeV;
4. compare the resulting fake factors directly to the note;
5. then switch only the ID definition back to medium to isolate the working-point effect.

That would answer whether the large numerical difference comes mainly from the ID/anti-ID definition, the event phase space, or the available samples.

### New validation scripts for the fake-normalisation audit

The remaining fake-estimate checks are now split into focused validation scripts
rather than being added to the main unfolding workflow. They are validation-only
studies and write under `outputs/validate_shadow_fakes/`.

| Question | Script | Output summary | Runtime mode |
|---|---|---|---|
| Does adding data-driven fakes double-count fake-like MC already present in the reconstructed MC stack? | `run/2017/validations/validate_preunfolding_stack_composition.py` | `outputs/validate_shadow_fakes/preunfolding_stack_composition/preunfolding_stack_composition_summary.md` | cache-only |
| Why do small `TauPt` fake factors produce a large total fake prediction? | `run/2017/validations/validate_failid_fake_application_breakdown.py` | `outputs/validate_shadow_fakes/failid_fake_application_breakdown/failid_fake_application_breakdown_summary.md` | cache-only |
| Can this framework reproduce the high-mass `tau + MET` note fake factors when using a Loose/VeryLoose-style definition? | `run/2017/validations/validate_note_like_loose_fake_factor.py` | `outputs/validate_shadow_fakes/note_like_loose_fake_factor/note_like_loose_fake_factor_summary.md` | first run may build a small cache |

The first script produces reconstructed-level stack-composition plots for
`no_shadow_bin` and `MTW_shadow_bin_250`. The key comparison is:

```text
all MC, no data-driven fakes
all MC + data-driven fakes
MC-contamination + data-driven fakes
fake-like MC only
```

If `MC-contamination + data-driven fakes` agrees better than `all MC + data-driven
fakes`, the fake estimate should be treated as replacing fake-like MC, not as
an additive component on top of it.

#### Pre-unfolding stack-composition result

The cache-only stack-composition validation has now been run:

```text
pixi run python run/2017/validations/validate_preunfolding_stack_composition.py
```

It wrote:

- summary: `outputs/validate_shadow_fakes/preunfolding_stack_composition/preunfolding_stack_composition_summary.md`
- plots:
  - `outputs/validate_shadow_fakes/preunfolding_stack_composition/plots/no_shadow_bin_preunfolding_stack_composition.png`
  - `outputs/validate_shadow_fakes/preunfolding_stack_composition/plots/MTW_shadow_bin_250_preunfolding_stack_composition.png`

The `tight_layout` warning printed by matplotlib is only a plotting-layout
warning; the summary and plots were produced.

| Configuration | Data | All MC, no fakes | All MC + fakes | MC contamination + fakes | Jet-fake-like MC only | Fakes | All MC/data | (All MC + fakes)/data | (MC contamination + fakes)/data |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_shadow_bin` | 1351.000 | 1359.868 | 1586.004 | 1472.324 | 113.679 | 226.136 | 1.007 | 1.174 | 1.090 |
| `MTW_shadow_bin_250` | 1428.000 | 1443.273 | 1687.088 | 1562.702 | 124.386 | 243.815 | 1.011 | 1.181 | 1.094 |

Interpretation:
the reconstructed MC stack without data-driven fakes already agrees with data
at about the `1%` level. Adding the current data-driven fake estimate on top of
all MC overshoots data by about `17-18%`, so an additive treatment double-counts
something fake-like. Replacing fake-like MC with the data-driven fake estimate
improves the comparison, but still overshoots by about `9%`.

This means fake-like MC double-counting is part of the bookkeeping problem, but
it is not the full explanation. The data-driven fake estimate itself is roughly
twice the size of the fake-like MC component in these reconstructed selections.
The next diagnostic should therefore be the fail-ID application breakdown:

```text
predicted fakes = FF(TauPt bin) * [SR fail-ID data - SR fail-ID MC-contamination]
```

split by prong and `TauPt` source bin. That test should identify whether the
large fake yield is driven by the SR fail-ID input population, the fake-factor
values, or the MC-contamination subtraction.

The second script makes the fake yield algebra visible source-bin by source-bin:

```text
predicted fakes in bin = FF(TauPt bin) * [SR fail-ID data - SR fail-ID MC-contamination]
```

This is the direct test of whether the large fake prediction is driven by the
transfer factor itself or by the fail-ID application population. It also reports
the available sample-level jet-fake-like/MC-contamination split in the SR anti-ID selection.

#### Fail-ID fake-application breakdown result

The fail-ID breakdown has now been run:

```text
pixi run python run/2017/validations/validate_failid_fake_application_breakdown.py
```

It wrote:

```text
outputs/validate_shadow_fakes/failid_fake_application_breakdown/failid_fake_application_breakdown_summary.md
```

The important result is that the large total fake prediction is not caused by
large fake-factor values. It is driven by the size of the SR fail-ID application
input, especially in the low-`TauPt` 1-prong bins.

| Configuration | Prong | Dominant source bins | Predicted fakes in dominant bins | Approx. share of prong prediction |
|---|---:|---|---:|---:|
| `no_shadow_bin` | 1 | `170-200`, `200-250` GeV | `81.736 + 68.002` | `0.709` |
| `no_shadow_bin` | 3 | `170-250` GeV | `2.465 + 3.908` | `0.419` |
| `MTW_shadow_bin_250` | 1 | `170-200`, `200-250` GeV | `96.188 + 69.212` | `0.727` |
| `MTW_shadow_bin_250` | 3 | `170-250` GeV | `3.118 + 4.461` | `0.459` |

For the no-shadow case, the 1-prong fake prediction is about `211` events and
the 3-prong prediction is about `15` events. For `MTW_shadow_bin_250`, the
corresponding numbers are about `227` and `16.5` events. This means the final
fake estimate is overwhelmingly a 1-prong low-`TauPt` subtraction.

The saved cache only contains the current framework `trueTau_...` MC-contamination split,
not a full truth-category split into hadronic tau, leptonic tau, electron, muon,
photon, and unmatched candidates. Within the available split, the SR fail-ID MC
fake-like component is mainly from `zll` and `wlnu`, while `wtaunu_had` is
almost entirely classified as MC-contamination in the fail-ID selections. For
example, in `MTW_shadow_bin_250`:

| Prong | Sample | All SR fail-ID MC | TrueTau/MC-contamination | Fake-like MC | Fake-like fraction |
|---:|---|---:|---:|---:|---:|
| 1 | `zll` | `562.501` | `203.565` | `358.936` | `0.638` |
| 1 | `wlnu` | `164.985` | `107.903` | `57.082` | `0.346` |
| 1 | `wtaunu_had` | `357.816` | `355.380` | `2.435` | `0.007` |
| 3 | `zll` | `280.960` | `106.252` | `174.708` | `0.622` |
| 3 | `wlnu` | `90.423` | `54.494` | `35.929` | `0.397` |
| 3 | `wtaunu_had` | `213.062` | `213.004` | `0.058` | `0.000` |

Interpretation:
the fake-factor values themselves are small and decreasing with `TauPt`, as
expected. The large fake subtraction comes from multiplying those small fake
factors by a large fail-ID application population. The most important bin
range for any future fake-model change is therefore `TauPt = 170-250` GeV in
the 1-prong fail-ID signal-region input.

Recommendation:
before making a nominal fake-model change, add or generate the full
truth-category split for the SR fail-ID application region. The current cache
can show sample-level jet-fake-like/MC-contamination behaviour, but it cannot yet answer
whether photons, leptonic tau/electron/muon candidates, or unmatched jets are
driving the relevant fail-ID population.

#### Fail-ID truth-category audit result

The category-level fail-ID audit has now been run:

```text
pixi run python run/2017/validations/validate_failid_truth_categories.py
```

It wrote:

- summary: `outputs/validate_shadow_fakes/failid_truth_categories/failid_truth_categories_summary.md`
- totals CSV: `outputs/validate_shadow_fakes/failid_truth_categories/failid_truth_category_totals.csv`
- source-bin CSV: `outputs/validate_shadow_fakes/failid_truth_categories/failid_truth_category_source_bins.csv`

This was a new event-loop run, but it is narrow: MC only, `TauPt` histograms
only, `no_shadow_bin` and `MTW_shadow_bin_250`, and only SR fail-ID selections.

The total SR fail-ID MC composition is:

| Configuration | Prong | Data fail-ID | All MC fail-ID | Hadronic tau | Electron | Photon | Jet-like/unmatched |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_shadow_bin` | 1 | `3148.000` | `1182.745` | `0.419` | `0.225` | `0.298` | `0.037` |
| `no_shadow_bin` | 3 | `1157.000` | `653.918` | `0.452` | `0.223` | `0.281` | `0.030` |
| `MTW_shadow_bin_250` | 1 | `3335.000` | `1258.185` | `0.421` | `0.223` | `0.298` | `0.037` |
| `MTW_shadow_bin_250` | 3 | `1198.000` | `683.725` | `0.450` | `0.223` | `0.281` | `0.031` |

The category composition in the low-`TauPt` source bins that dominate the fake
prediction, `170-250 GeV`, is:

| Configuration | Prong | Total MC in `170-250` | Hadronic tau | Photon | Electron | Jet-like/unmatched |
|---|---:|---:|---:|---:|---:|---:|
| `no_shadow_bin` | 1 | `816.453` | `0.410` | `0.305` | `0.226` | `0.037` |
| `no_shadow_bin` | 3 | `350.598` | `0.459` | `0.298` | `0.196` | `0.030` |
| `MTW_shadow_bin_250` | 1 | `887.151` | `0.413` | `0.306` | `0.222` | `0.036` |
| `MTW_shadow_bin_250` | 3 | `374.966` | `0.452` | `0.298` | `0.200` | `0.032` |

Interpretation:
the fail-ID MC in the application region is not mostly conventional
jet-like/unmatched candidates. It is dominated by hadronic-tau, photon, and
electron matched candidates. The jet-like/unmatched category is only about
`3-4%` of the MC fail-ID population.

This matters because the current framework `trueTau_...` subtraction removes
matched hadronic tau, electron, and muon candidates:

```text
MatchedTruthParticle_isHadronicTau == true
|| MatchedTruthParticle_isMuon == true
|| MatchedTruthParticle_isElectron == true
```

It does not remove photon-matched candidates, and the photon-matched component
is large: about `28-31%` of the fail-ID MC in the relevant `TauPt` bins. If the
analysis concept is "subtract real/prompt non-jet backgrounds from the anti-ID
data before estimating jet fakes", photon-matched candidates may need explicit
treatment rather than being folded into the data-driven fake estimate by
default.

This does not by itself solve the full normalisation problem. The data fail-ID
yield is much larger than the summed MC fail-ID yield, so even subtracting
additional MC categories cannot make the data-driven estimate disappear.
However, the audit identifies a concrete definition choice that should be
checked against the intended fake-factor method:

```text
Should photon-matched tau candidates be part of the MC-contamination subtraction,
or should they be treated as part of the data-driven fake estimate?
```

Recommendation:
the next code-level test should be a validation-only fake estimate with
`trueTau_...` replaced by an expanded "MC" subtraction that also
includes photons, and possibly leptonic-tau matches. Compare the resulting
pre-unfolding stack and fake yield to the current baseline. This is a much more
targeted test than another unfolding rerun.

#### Photon-expanded MC-contamination subtraction test

The photon-expanded subtraction test has now been run:

```text
pixi run python run/2017/validations/validate_photon_nonfake_subtraction.py
```

It wrote:

- summary: `outputs/validate_shadow_fakes/photon_nonfake_subtraction/photon_nonfake_subtraction_summary.md`
- CSV: `outputs/validate_shadow_fakes/photon_nonfake_subtraction/photon_nonfake_subtraction_summary.csv`
- cache: `outputs/validate_shadow_fakes/photon_nonfake_subtraction/root/validate_photon_nonfake_subtraction.root`

This was a new event-loop validation, not a nominal production change. It
compares three MC-contamination-subtraction definitions:

| Model | MC-contamination subtraction definition |
|---|---|
| `current` | hadronic tau, electron, muon |
| `with_photon` | current plus photon-matched candidates |
| `with_photon_leptonic_tau` | current plus photon and leptonic-tau candidates |

The resulting inclusive fake yields and stack ratios are:

| Configuration | Model | Data pass-ID | All MC pass-ID | MC-contamination pass-ID | Fakes | All MC + fakes / data | MC-contamination + fakes / data |
|---|---|---:|---:|---:|---:|---:|---:|
| `no_shadow_bin` | `current` | `1351.000` | `1359.867` | `1246.190` | `226.136` | `1.174` | `1.090` |
| `no_shadow_bin` | `with_photon` | `1351.000` | `1359.867` | `1347.350` | `182.380` | `1.142` | `1.132` |
| `no_shadow_bin` | `with_photon_leptonic_tau` | `1351.000` | `1359.867` | `1350.060` | `181.381` | `1.141` | `1.134` |
| `MTW_shadow_bin_250` | `current` | `1351.000` | `1359.867` | `1246.190` | `226.136` | `1.174` | `1.090` |
| `MTW_shadow_bin_250` | `with_photon` | `1351.000` | `1359.867` | `1347.350` | `182.380` | `1.142` | `1.132` |
| `MTW_shadow_bin_250` | `with_photon_leptonic_tau` | `1351.000` | `1359.867` | `1350.060` | `181.381` | `1.141` | `1.134` |

Per-prong fake yields show where the change enters:

| Configuration | Prong | Model | Fake yield |
|---|---:|---|---:|
| `no_shadow_bin` | 1 | `current` | `210.924` |
| `no_shadow_bin` | 1 | `with_photon` | `175.809` |
| `no_shadow_bin` | 1 | `with_photon_leptonic_tau` | `175.001` |
| `no_shadow_bin` | 3 | `current` | `15.212` |
| `no_shadow_bin` | 3 | `with_photon` | `6.571` |
| `no_shadow_bin` | 3 | `with_photon_leptonic_tau` | `6.380` |

The `MTW_shadow_bin_250` rows give the same integrated yields in this test
because the measured pass-ID/fail-ID phase space is the same for the selected
production configuration; the shadow-bin difference enters the response, not
this inclusive fake-yield diagnostic.

Interpretation:
including photon-matched candidates in the MC-contamination subtraction has a real but
limited effect. It reduces the total fake prediction from about `226` to about
`182` events, a reduction of roughly `19%`. Most of that reduction is in the
dominant 1-prong component, but the relative reduction is larger for 3-prong
(`15.2` to `6.6` events).

This does **not** solve the pre-unfolding normalisation issue. If the stack is
formed as `all MC + data-driven fakes`, the ratio improves from `1.174` to
`1.142`, but the stack still over-predicts data. If the stack is formed as
`MC-contamination + data-driven fakes`, moving photon candidates into the MC-contamination
definition actually raises the ratio from `1.090` to about `1.132`, because the
ID-region photon contribution added to the MC-contamination stack is larger than the fake
yield reduction.

Recommendation:
do not promote photon-expanded subtraction to the nominal fake model yet. It is
a plausible definition systematic, and it confirms that the current fake-like
category is not purely jet-like. However, the main issue remains broader than
photon treatment alone: the baseline all-MC stack is already approximately at
the data yield, so adding any sizeable data-driven fake component tends to
over-count unless the corresponding fake-like MC component is removed or the
fake-factor method is redefined more carefully.

The third script is the like-for-like comparison requested after reading the
high-mass `tau + MET` note. It uses `TauLooseWP == 1` for the numerator and
`TauLooseWP == 0 && TauRNNJetScore > 0.01` as the available VeryLoose-style
anti-ID proxy. Because no exact VeryLoose branch has been found in the current
ntuples, this remains an approximation. The comparison is still useful: if the
Loose-proxy result approaches Table 14 while the medium-reference result stays
small, the large numerical difference is mainly an ID/anti-ID working-point
effect.

#### Note-like Loose fake-factor comparison result

The note-like Loose/VeryLoose-proxy validation has now been run:

```text
pixi run python run/2017/validations/validate_note_like_loose_fake_factor.py
```

It wrote:

```text
outputs/validate_shadow_fakes/note_like_loose_fake_factor/note_like_loose_fake_factor_summary.md
```

This run did build its own small validation cache under:

```text
outputs/validate_shadow_fakes/note_like_loose_fake_factor/root/
```

The comparison to Table 14 of the high-mass `tau + MET` note is:

| ID model | Prong | `TauPt` bin [GeV] | This validation FF | ATLAS note FF | This / ATLAS note |
|---|---:|---:|---:|---:|---:|
| Loose numerator / fail Loose plus `TauRNNJetScore > 0.01` proxy denominator | 1 | 350-500 | `0.10322` | `0.362` | `0.285` |
| Loose numerator / fail Loose plus `TauRNNJetScore > 0.01` proxy denominator | 1 | 500-1000 | `0.07020` | `0.336` | `0.209` |
| Loose numerator / fail Loose plus `TauRNNJetScore > 0.01` proxy denominator | 3 | 350-500 | `0.03152` | `0.116` | `0.272` |
| Loose numerator / fail Loose plus `TauRNNJetScore > 0.01` proxy denominator | 3 | 500-1000 | `0.02063` | `0.083` | `0.249` |

The same phase-space test with medium-ID definitions gives:

| ID model | Prong | `TauPt` bin [GeV] | Fake factor |
|---|---:|---:|---:|
| Medium numerator / fail Medium plus `TauRNNJetScore > 0.01` denominator | 1 | 350-500 | `0.04844` |
| Medium numerator / fail Medium plus `TauRNNJetScore > 0.01` denominator | 1 | 500-1000 | `0.03232` |
| Medium numerator / fail Medium plus `TauRNNJetScore > 0.01` denominator | 3 | 350-500 | `0.01102` |
| Medium numerator / fail Medium plus `TauRNNJetScore > 0.01` denominator | 3 | 500-1000 | `0.00760` |

Interpretation:
switching from medium-ID to a Loose-proxy definition increases the fake factors
by roughly a factor of `2-3`, which is the expected direction. However, the
Loose-proxy fake factors are still only about `20-30%` of the ATLAS note values.
The difference is therefore not explained by the medium-vs-loose working point
alone. The remaining difference could come from the lack of an exact VeryLoose
denominator branch, the event phase space, trigger/year/sample composition, or
the note's multijet-enrichment details that are not exactly reproduced here.

Recommendation:
do not use the ATLAS note values as a direct numerical target for this analysis.
Use them as qualitative support for the method structure: prong-split,
`TauPt`-dependent fake factors and explicit transfer/shape uncertainties. The
next useful validation is not another high-`TauPt` fake-factor comparison; it is
the full truth-category audit of the fail-ID application region.

### Proposed low-MET fake-enriched validation experiment

The next fake-estimate test should be a deliberately low-MET, fake-enriched derivation region. This is not yet a proposed nominal change to the unfolding. It is a targeted validation experiment to decide whether the current fake-factor control region is too close to the unstable upper-MET sideband.

The motivation is the current validation pattern:

- the nominal `TauPt` fake factor transfers well across MTW sidebands;
- it transfers poorly across MET sidebands, especially for 3-prong candidates;
- lowering the simple derivation cut from `MET < 170` to `MET < 120` does not, by itself, improve the same validation target;
- the ATLAS high-mass `tau + MET` search derives its jet-to-tau fake factors in a more explicitly multijet-enriched low-MET region, then validates and assigns transfer uncertainties.

The supporting sources are:

- local technical note: `/home/keanu/Uni_Stuff_Queen_Mary/Reading list/High-mass resonances to taunu may 21 NOTE.pdf`, especially Sec. 6.1, Sec. 7.1.5, and Sec. 7.1.8;
- ATLAS Collaboration, "Search for high-mass resonances in final states with a tau-lepton and missing transverse momentum with the ATLAS detector", Phys. Rev. D 109 (2024) 112008, arXiv:2402.16576: https://arxiv.org/abs/2402.16576

In the high-mass `tau + MET` note, the fake-factor method is structured as:

```text
MJ-ID:
  low-MET, multijet-enriched region
  tau passes the tau-ID working point

MJ-nonID:
  same low-MET, multijet-enriched region
  tau fails ID but passes a looser anti-ID floor

anti-ID application region:
  signal-like selection
  tau fails ID but passes the looser anti-ID floor
```

The measured transfer factor is:

```text
FF(pT, Ntracks) =
  (data MJ-ID - MC-contamination MJ-ID)
  /
  (data MJ-nonID - MC-contamination MJ-nonID)
```

and it is applied to anti-ID events after subtracting MC-contamination. The note uses `MET < 100 GeV` for the nominal multijet fake-factor region and studies alternative MET windows such as `[30,100]`, `[50,100]`, `[70,100]`, and `[0,150]` as systematic variations. It also validates the low-MET fake factors in a high-MET alternative validation region.

For this analysis, the closest practical validation should use the same principle while preserving our own signal definition and medium-ID working point:

```text
Fake-factor derivation numerator:
  passReco == 1
  TauBaselineWP == 1
  abs(TauCharge) == 1
  badJet == 0
  pass eta
  TauPt > 170
  MET < 100
  pass medium ID
  split 1-prong and 3-prong

Fake-factor derivation denominator:
  same region
  fail medium ID
  TauBDTEleScore > 0.1
  TauRNNJetScore > 0.01
  split 1-prong and 3-prong
```

The first validation target should stay outside the final `MET > 170` signal region:

```text
MTW >= 350
MET < 170
pass/fail medium ID
split 1-prong and 3-prong
```

This answers a clean question: if the fake factor is derived in a more fake-enriched low-MET region, does it predict the pass-ID residual in a nominal-MTW, still-control-MET region better than the current derivation?

A second target can then be the already-used signal-like proxy:

```text
MTW >= 350
MET > 170
fail-ID input and pass-ID residual comparison
```

That second target is closer to the final unfolding input, but it is less independent, so it should not be the only validation criterion.

This is not a literal copy of the high-mass note. The differences are important:

1. the note uses Loose/VeryLoose tau-ID definitions, while this analysis uses medium ID and a fail-medium anti-ID floor;
2. the note has a different search-region definition, including a tau-`pT`/`MET` balance requirement that is not part of this unfolding selection in the same way;
3. the note ultimately builds a high-`MTW` search background with an explicit tail extrapolation, while this analysis needs a stable subtraction before fiducial unfolding;
4. this analysis must preserve the `W -> tau nu -> hadrons` truth definition and the nonfiducial signal subtraction already needed for unfolding closure.

The validation should therefore test the transferable idea, not import the full search method:

```text
derive fake factors in a cleaner low-MET fake-enriched region,
validate the transfer to nominal-MTW and signal-like regions,
then treat any residual difference as a transfer systematic unless the validation strongly supports changing the nominal fake model.
```

Recommended implementation:

```text
run/2017/validations/validate_low_met_fake_region.py
```

with outputs cached under:

```text
outputs/validate_shadow_fakes/low_met_fake_region/
```

The script should be intentionally narrow:

- `MTW` only;
- `MTW_shadow_bin_300` only for the smoke test;
- medium ID only;
- 1-prong and 3-prong split only;
- no systematics;
- no unfolding;
- read cached analysis outputs where possible, and only run new ROOT loops for the genuinely new `MET < 100` derivation selections.

The output table should compare the current method and the low-MET fake-enriched method:

| Check | Required output |
| --- | --- |
| CR health | numerator, denominator, negative numerator bins, tiny/non-positive denominator bins |
| Transfer result | validation fail-ID input, predicted fake yield, pass-ID `data - MC-contamination` target |
| Ratio | prediction / target for 1-prong, 3-prong, and combined |
| Shape | `MTW` closure plot for current vs low-MET derivation |
| Systematic candidate | difference between current and low-MET predictions |

The success criteria are:

1. the low-MET numerator and denominator are positive and statistically stable in the `TauPt`/prong bins;
2. the validation prediction is closer to the pass-ID `data - MC-contamination` target than the current `MET < 170` fake factor;
3. the 3-prong prediction is not driven by tiny denominator bins or negative MC-contamination-subtracted bins;
4. the predicted `MTW` shape is not visibly sculpted by the control-region choice;
5. the difference from the nominal fake estimate can be turned into a defensible transfer uncertainty if the low-MET method is not adopted as the central model.

### Low-MET fake-enriched validation result

The low-MET fake-enriched validation was implemented in:

```text
run/2017/validations/validate_low_met_fake_region.py
```

and run once with new ROOT event loops. The outputs are:

```text
outputs/validate_shadow_fakes/low_met_fake_region/
```

The summary table is:

```text
outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md
```

The cached histogram file for later reruns is:

```text
outputs/validate_shadow_fakes/low_met_fake_region/root/validate_low_met_fake_region.root
```

The test compares two fake-factor derivation methods:

1. the current MTW-shadow control region, `300 <= MTW < 350` and `MET < 170`;
2. the low-MET fake-enriched region, `MET < 100`.

Both are applied to the same validation targets. The first target, `MTW >= 350` and `MET < 170`, is the cleaner independent validation because it stays outside the final high-MET signal region. The second target, `MTW >= 350` and `MET >= 170`, is a signal-like proxy and is therefore closer to the unfolding input, but less independent.

| Target | Prong | Method | CR numerator | CR denominator | Problem bins | Predicted fakes | Validation target | Prediction / target |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MTW >= 350`, `MET < 170` | 1 | current MTW-shadow CR | 1778.877 | 24336.715 | 0 neg num, 0 tiny den | 699.203 | 677.359 | 1.032 |
| `MTW >= 350`, `MET < 170` | 1 | low-MET fake-enriched | 326482.057 | 2990296.991 | 0 neg num, 3 tiny den | 681.409 | 677.359 | 1.006 |
| `MTW >= 350`, `MET < 170` | 3 | current MTW-shadow CR | 253.202 | 10786.827 | 1 neg num, 0 tiny den | 128.217 | 124.117 | 1.033 |
| `MTW >= 350`, `MET < 170` | 3 | low-MET fake-enriched | 38779.324 | 703576.617 | 0 neg num, 1 tiny den | 122.483 | 124.117 | 0.987 |
| `MTW >= 350`, `MET >= 170` | 1 | current MTW-shadow CR | 1778.877 | 24336.715 | 0 neg num, 0 tiny den | 203.797 | 163.335 | 1.248 |
| `MTW >= 350`, `MET >= 170` | 1 | low-MET fake-enriched | 326482.057 | 2990296.991 | 0 neg num, 3 tiny den | 210.924 | 163.335 | 1.291 |
| `MTW >= 350`, `MET >= 170` | 3 | current MTW-shadow CR | 253.202 | 10786.827 | 1 neg num, 0 tiny den | 13.567 | -58.523 | -0.232 |
| `MTW >= 350`, `MET >= 170` | 3 | low-MET fake-enriched | 38779.324 | 703576.617 | 0 neg num, 1 tiny den | 15.212 | -58.523 | -0.260 |

The independent control-MET target gives the most useful result. In that region, the low-MET fake-enriched derivation improves both prongs:

```text
1-prong:  prediction / target improves from 1.032 to 1.006
3-prong:  prediction / target improves from 1.033 to 0.987
combined: prediction / target improves from about 1.032 to about 1.003
```

This is the first validation result that supports the ATLAS-like low-MET fake-enriched direction. It does not merely change the MET threshold; it derives the fake factor in a much higher-statistics fake-enriched region. That removes the negative 3-prong numerator bin seen in the current MTW-shadow CR and gives a more stable denominator. This is consistent with the ATLAS high-mass `tau + MET` strategy: derive the fake factor in a low-MET multijet-enriched region, then validate the extrapolation and assign uncertainties.

The signal-like high-MET proxy does **not** improve. The 1-prong fake prediction remains high, and the 3-prong pass-ID `data - MC-contamination` target is negative. That negative target means the high-MET 3-prong proxy is not a clean validation of the fake factor alone; it is also sensitive to the MC-contamination subtraction, especially the `wtaunu_had` component already identified as problematic in the prong-balance checks.

Representative plots:

| Control-MET target | Signal-like proxy |
| --- | --- |
| <img src="outputs/validate_shadow_fakes/low_met_fake_region/plots/low_met_fake_region/MTW_shadow_bin_300_nominal_mtw_control_metlt170_1prong_low_met_fake_region.png" width="390"> | <img src="outputs/validate_shadow_fakes/low_met_fake_region/plots/low_met_fake_region/MTW_shadow_bin_300_signal_like_metgt170_1prong_low_met_fake_region.png" width="390"> |
| <img src="outputs/validate_shadow_fakes/low_met_fake_region/plots/low_met_fake_region/MTW_shadow_bin_300_nominal_mtw_control_metlt170_3prong_low_met_fake_region.png" width="390"> | <img src="outputs/validate_shadow_fakes/low_met_fake_region/plots/low_met_fake_region/MTW_shadow_bin_300_signal_like_metgt170_3prong_low_met_fake_region.png" width="390"> |

Interpretation:

- The low-MET fake-enriched derivation is better than the current derivation in the independent `MTW >= 350`, `MET < 170` validation target.
- It should not yet be adopted as a nominal fake model for the unfolding, because the signal-like high-MET proxy is still not closed.
- The high-MET proxy failure is not purely a fake-factor problem; the negative 3-prong target shows that the pass-ID MC-contamination subtraction is also entering.
- The result supports using the low-MET derivation as the next candidate fake model to test in the main closure workflow, with the difference relative to the current method treated as a transfer/extrapolation uncertainty unless a stronger high-MET validation can be made.

Recommended next steps:

1. add a switch in `analysis_shadow_unfold.py` to run the low-MET fake-enriched derivation as an alternative fake model, not yet as the only nominal model;
2. rerun the main closure comparison for `MTW_shadow_bin_300` only, using cached inputs where possible;
3. compare unfolded data with current fakes, low-MET fakes, and no fakes;
4. keep the signal-like 3-prong proxy caveat explicit in the thesis write-up.

### ATLAS-like high-MET fake-transfer validation

Question:
Can the low-MET fake-enriched fake-factor strategy used in the ATLAS high-mass `tau + MET` search transfer into a high-MET validation region closer to their alternative validation region?

Implementation:
- script: `run/2017/validations/validate_atlas_like_fake_transfer.py`
- outputs: `outputs/validate_shadow_fakes/atlas_like_fake_transfer/`
- summary: `outputs/validate_shadow_fakes/atlas_like_fake_transfer/atlas_like_fake_transfer_summary.md`
- cache: `outputs/validate_shadow_fakes/atlas_like_fake_transfer/root/validate_atlas_like_fake_transfer.root`
- mode: new ROOT event loops were run once; reruns should be cache-only

The script derives prong-split `TauPt` fake factors in low-MET fake-enriched regions:

```text
0 <= MET < 100
30 <= MET < 100
50 <= MET < 100
70 <= MET < 100
0 <= MET < 150
```

and applies them to high-MET imbalanced validation targets:

```text
MTW >= 240, MET >= 170, TauPt / MET < 0.7
MTW >= 350, MET >= 170, TauPt / MET < 0.7
```

Important technical caveat:
The `MTW` histogram binning used by this analysis starts at `350 GeV`:

```text
[350, 375, 400, 430, 465, 500, 550, 600, 700, 850, 1000, 2000]
```

Therefore, the `MTW >= 240` and `MTW >= 350` rows have the same `MTW`-histogram target integrals in this output. The `MTW >= 240` selection is applied at event level, but the resulting `MTW` histogram cannot display or integrate the `240-350 GeV` part of the validation region. This means the current run is still useful for testing the high-MET imbalanced population **inside the nominal unfolded MTW range**, but it is not yet a complete reproduction of the ATLAS-style `MTW > 240 GeV` validation region. A true reproduction needs a validation-specific `MTW` binning that starts below `350 GeV`.

Compact result:

| Validation target | Prong | MET-window spread | Prediction / target |
| --- | ---: | --- | ---: |
| high-MET imbalanced, nominal MTW bins | 1 | `0 <= MET < 100` to `0 <= MET < 150` | `0.660-0.683` |
| high-MET imbalanced, nominal MTW bins | 3 | `0 <= MET < 100` to `0 <= MET < 150` | `0.215-0.220` |
| high-MET imbalanced, nominal MTW bins | combined | `0 <= MET < 100` to `0 <= MET < 150` | `0.538-0.556` |

Representative rows from the summary:

| MET window | Prong | CR numerator | CR denominator | Problem bins | Predicted fakes | Validation target | Prediction / target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0 <= MET < 100` | 1 | 326482.055 | 2990296.991 | 0 neg num, 3 tiny den | 8.059 | 11.817 | 0.682 |
| `70 <= MET < 100` | 1 | 17267.916 | 173139.106 | 0 neg num, 2 tiny den | 7.795 | 11.817 | 0.660 |
| `0 <= MET < 150` | 1 | 330160.575 | 3022820.504 | 0 neg num, 3 tiny den | 8.072 | 11.817 | 0.683 |
| `0 <= MET < 100` | 3 | 38779.324 | 703576.617 | 0 neg num, 1 tiny den | 0.958 | 4.429 | 0.216 |
| `70 <= MET < 100` | 3 | 2200.863 | 48901.341 | 0 neg num, 0 tiny den | 0.975 | 4.429 | 0.220 |
| `0 <= MET < 150` | 3 | 39145.866 | 716173.416 | 0 neg num, 0 tiny den | 0.956 | 4.429 | 0.216 |

The most important feature is that the MET-window envelope is very small. The five low-MET windows produce almost identical fake predictions. That means the ATLAS-style MET-window variations, at least in this implementation and binning, do not cover the discrepancy in the high-MET imbalanced validation region.

The 1-prong prediction undershoots the high-MET imbalanced target by about `30-35%`. The 3-prong prediction undershoots by roughly a factor of five. This is the opposite direction from the earlier signal-like high-MET pass-ID proxy, where the fake subtraction looked too large once compared with the total unfolding budget. The difference matters: this imbalanced validation region is not the nominal SR, and the absolute yields are small. It is best interpreted as a transfer/composition stress test, not as a direct correction factor for the nominal analysis.

Representative plots:

| 1-prong high-MET imbalanced | 3-prong high-MET imbalanced |
| --- | --- |
| <img src="outputs/validate_shadow_fakes/atlas_like_fake_transfer/plots/atlas_like_fake_transfer/atlas_like_low_met_atlas_like_high_met_imbalanced_1prong_atlas_like_fake_transfer.png" width="390"> | <img src="outputs/validate_shadow_fakes/atlas_like_fake_transfer/plots/atlas_like_fake_transfer/atlas_like_low_met_atlas_like_high_met_imbalanced_3prong_atlas_like_fake_transfer.png" width="390"> |

Interpretation:

- The low-MET fake-factor numerator and denominator are healthy in this test; there are no negative numerator bins.
- Varying the low-MET window barely changes the predicted high-MET imbalanced fake yield.
- This validation therefore does **not** support the idea that a simple MET-window envelope is enough to cover the fake-transfer problem.
- The failure is stronger for 3-prong candidates, again pointing towards fake-source composition or MC-contamination-subtraction modelling rather than just a bad MET threshold.
- Because the nominal `MTW` bins start at `350 GeV`, a proper ATLAS-like `MTW > 240 GeV` validation still needs a validation-specific binning before this can be used as a thesis-quality reproduction of their validation-region logic.

Recommendation:

Do not promote this ATLAS-like low-MET model to the nominal fake estimate yet. The useful conclusion is narrower:

1. low-MET fake-factor derivation is statistically healthy;
2. MET-window variations alone do not explain the high-MET transfer mismatch;
3. the next validation should either add a custom `MTW` binning down to `240 GeV` for the ATLAS-like region, or move to the fake-source composition handle, i.e. a read-only audit for tau seed-width / tau-width variables.

### Tau-width fake-composition validation

Question:
Does the low-MET fake-factor determination region have the same tau-width shape as the high-MET anti-ID application region, or is there evidence for fake-source composition mismatch?

Implementation:
- script: `run/2017/validations/validate_tau_width_composition.py`
- outputs: `outputs/validate_shadow_fakes/tau_width_composition/`
- summary: `outputs/validate_shadow_fakes/tau_width_composition/tau_width_composition_summary.md`
- cache: `outputs/validate_shadow_fakes/tau_width_composition/root/validate_tau_width_composition.root`
- mode: new ROOT event loops were run once; reruns should be cache-only

The validation uses tau track-width branches found in the input ntuples:

```text
TauTrackWidthPt1000PV
TauTrackWidthPt500PV
TauTrackWidthPt1000TV
TauTrackWidthPt500TV
```

These are not exactly named "tau jet seed width", but they are plausible width-like proxies for the same underlying issue: whether the fake-like tau candidates are narrow or broad, which is sensitive to fake-source composition. In the ATLAS high-mass `tau + MET` analysis, tau jet seed width is used to assess quark/gluon composition differences between the multijet fake-factor region and the anti-ID application region.

The comparison is made on fake-like histograms:

```text
data - MC-contamination
```

for:

- low-MET fake-factor denominator, `MET < 100`;
- nominal high-MET anti-ID application region, `MTW >= 350`, `MET >= 170`;
- ATLAS-like high-MET imbalanced anti-ID region, `MTW >= 350`, `MET >= 170`, `TauPt / MET < 0.7`.

The cleanest comparison is the nominal high-MET anti-ID region, because it has much larger yield and far fewer negative bins than the imbalanced validation region.

Key rows:

| Variable | Prong | Comparison | Low-MET fake-like yield | Target fake-like yield | Low-MET mean | Target mean | Relative mean shift | L1 shape distance |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `TauTrackWidthPt1000PV` | 1 | nominal high-MET anti-ID | 2977426.152 | 1990.266 | 0.02325 | 0.04404 | 0.894 | 0.134 |
| `TauTrackWidthPt500PV` | 1 | nominal high-MET anti-ID | 2982159.157 | 2121.601 | 0.02623 | 0.06031 | 1.299 | 0.176 |
| `TauTrackWidthPt1000TV` | 1 | nominal high-MET anti-ID | 2971979.477 | 2231.217 | 0.02322 | 0.03773 | 0.625 | 0.166 |
| `TauTrackWidthPt500TV` | 1 | nominal high-MET anti-ID | 2976933.974 | 2264.718 | 0.02623 | 0.04524 | 0.725 | 0.182 |
| `TauTrackWidthPt1000PV` | 3 | nominal high-MET anti-ID | 703499.007 | 715.962 | 0.01568 | 0.02002 | 0.277 | 0.131 |
| `TauTrackWidthPt500PV` | 3 | nominal high-MET anti-ID | 703600.854 | 718.103 | 0.01667 | 0.02149 | 0.289 | 0.131 |
| `TauTrackWidthPt1000TV` | 3 | nominal high-MET anti-ID | 702422.088 | 720.750 | 0.01570 | 0.02036 | 0.297 | 0.134 |
| `TauTrackWidthPt500TV` | 3 | nominal high-MET anti-ID | 702691.866 | 720.272 | 0.01674 | 0.02107 | 0.258 | 0.130 |

The high-MET anti-ID fake-like population is consistently wider than the low-MET fake-factor denominator. The effect is strongest for 1-prong candidates:

- `TauTrackWidthPt1000PV`: mean width increases by about `89%`;
- `TauTrackWidthPt500PV`: mean width increases by about `130%`;
- `TauTrackWidthPt1000TV`: mean width increases by about `63%`;
- `TauTrackWidthPt500TV`: mean width increases by about `73%`.

The 3-prong shifts are smaller but still systematic, around `26-30%`. This supports the hypothesis that the low-MET fake-factor determination region and high-MET anti-ID application region have different fake-like tau compositions.

The ATLAS-like high-MET imbalanced region shows even larger shape distances in some rows, especially 3-prong, but its fake-like yield is very small and many bins become negative after MC-contamination subtraction. Those rows should be treated as qualitative only.

Representative plots:

| 1-prong `TauTrackWidthPt1000PV` | 3-prong `TauTrackWidthPt1000PV` |
| --- | --- |
| <img src="outputs/validate_shadow_fakes/tau_width_composition/plots/tau_width_composition/TauTrackWidthPt1000PV_1prong_tau_width_composition.png" width="390"> | <img src="outputs/validate_shadow_fakes/tau_width_composition/plots/tau_width_composition/TauTrackWidthPt1000PV_3prong_tau_width_composition.png" width="390"> |

Interpretation:

- This is the first direct evidence, in our own ntuples, for a fake-source/composition mismatch between the low-MET fake-factor denominator and the high-MET anti-ID application region.
- The result is qualitatively aligned with the ATLAS tau+MET treatment: a width-like tau variable can diagnose differences between the fake-factor measurement region and the application region.
- This does not by itself define a correction, because `data - MC-contamination` subtraction in sparse high-MET regions can produce negative bins and the branch is a track-width proxy rather than the exact ATLAS seed-width observable.
- It does justify treating fake-source composition as a real systematic candidate rather than a speculative explanation.

Recommendation:

Do not immediately reweight the nominal fake estimate. The next defensible validation step is to build a width-reweighting diagnostic:

1. derive a low-MET-to-high-MET anti-ID width reweighting using the highest-statistics comparison, preferably nominal high-MET anti-ID rather than the sparse imbalanced region;
2. apply that reweighting to the low-MET fake-factor denominator or to the fake-factor prediction as a systematic variation;
3. compare the shifted fake prediction to the existing low-MET and current fake predictions;
4. if the effect is large and stable across `TauTrackWidthPt1000PV` and `TauTrackWidthPt500PV`, treat it as a fake-source composition uncertainty.

### Tau-width reweighting diagnostic

Question:
If the high-MET anti-ID application region has a different tau-width shape from the low-MET fake-factor denominator, does an ATLAS-like width reweighting move the fake prediction in the expected direction?

Implementation:
- script: `run/2017/validations/validate_tau_width_reweighting.py`
- outputs: `outputs/validate_shadow_fakes/tau_width_reweighting/`
- summary: `outputs/validate_shadow_fakes/tau_width_reweighting/tau_width_reweighting_summary.md`
- cache: `outputs/validate_shadow_fakes/tau_width_reweighting/root/validate_tau_width_reweighting_multiwidth.root`
- mode: new ROOT event loops were run once; later reruns should use the cached ROOT file
- width proxies: `TauTrackWidthPt1000PV`, `TauTrackWidthPt500PV`
- fake-factor source variable: `TauPt`
- target observable: `MTW`
- width weights are uncapped; the dedicated cap scan below found no sparse-bin instability requiring a cap

This is a diagnostic, not a proposed nominal correction. It tests two directions:

- `application_to_lowmet`: reweight the high-MET anti-ID application events so their tau-width shape looks more like the low-MET fake-factor denominator. This is the ATLAS-like systematic direction.
- `lowmet_to_application`: the opposite stress test, upweighting the broader high-MET-like width tail. This is diagnostic only and should not be used as the nominal correction.

Result:

| Width variable | Prong | Prediction | Integral | Target integral | Prediction / target | Relative to nominal fake prediction |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `TauTrackWidthPt1000PV` | 1 | nominal low-MET fake factor | 210.924 | 163.335 | 1.291 | 1.000 |
| `TauTrackWidthPt1000PV` | 1 | `application_to_lowmet` width reweight | 167.680 | 163.335 | 1.027 | 0.795 |
| `TauTrackWidthPt1000PV` | 1 | `lowmet_to_application` width reweight | 247.168 | 163.335 | 1.513 | 1.172 |
| `TauTrackWidthPt500PV` | 1 | nominal low-MET fake factor | 210.924 | 163.335 | 1.291 | 1.000 |
| `TauTrackWidthPt500PV` | 1 | `application_to_lowmet` width reweight | 178.643 | 163.335 | 1.094 | 0.847 |
| `TauTrackWidthPt500PV` | 1 | `lowmet_to_application` width reweight | 305.434 | 163.335 | 1.870 | 1.448 |
| `TauTrackWidthPt1000PV` | 3 | nominal low-MET fake factor | 15.212 | -58.523 | -0.260 | 1.000 |
| `TauTrackWidthPt1000PV` | 3 | `application_to_lowmet` width reweight | 15.135 | -58.523 | -0.259 | 0.995 |
| `TauTrackWidthPt1000PV` | 3 | `lowmet_to_application` width reweight | 18.199 | -58.523 | -0.311 | 1.196 |
| `TauTrackWidthPt500PV` | 3 | nominal low-MET fake factor | 15.212 | -58.523 | -0.260 | 1.000 |
| `TauTrackWidthPt500PV` | 3 | `application_to_lowmet` width reweight | 15.537 | -58.523 | -0.265 | 1.021 |
| `TauTrackWidthPt500PV` | 3 | `lowmet_to_application` width reweight | 17.864 | -58.523 | -0.305 | 1.174 |

Representative plots:

| 1-prong `TauTrackWidthPt1000PV` | 1-prong `TauTrackWidthPt500PV` |
| --- | --- |
| <img src="outputs/validate_shadow_fakes/tau_width_reweighting/plots/tau_width_reweighting/TauTrackWidthPt1000PV_1prong_tau_width_reweighting.png" width="390"> | <img src="outputs/validate_shadow_fakes/tau_width_reweighting/plots/tau_width_reweighting/TauTrackWidthPt500PV_1prong_tau_width_reweighting.png" width="390"> |

| 3-prong `TauTrackWidthPt1000PV` | 3-prong `TauTrackWidthPt500PV` |
| --- | --- |
| <img src="outputs/validate_shadow_fakes/tau_width_reweighting/plots/tau_width_reweighting/TauTrackWidthPt1000PV_3prong_tau_width_reweighting.png" width="390"> | <img src="outputs/validate_shadow_fakes/tau_width_reweighting/plots/tau_width_reweighting/TauTrackWidthPt500PV_3prong_tau_width_reweighting.png" width="390"> |

Interpretation:

For 1-prong candidates, both width proxies move in the expected direction if the fake-source composition mismatch is real. The nominal low-MET fake factor overpredicts the high-MET pass-ID target by about `29%`. Reweighting the high-MET anti-ID application region toward the low-MET width shape reduces the prediction:

- `TauTrackWidthPt1000PV`: prediction/target improves from `1.291` to `1.027`;
- `TauTrackWidthPt500PV`: prediction/target improves from `1.291` to `1.094`.

The opposite stress-test direction increases the prediction and worsens the agreement:

- `TauTrackWidthPt1000PV`: prediction/target becomes `1.513`;
- `TauTrackWidthPt500PV`: prediction/target becomes `1.870`.

The two proxies do not give identical shifts, but they agree on the qualitative direction: the broader high-MET anti-ID width shape is associated with a larger fake prediction, and correcting the application region back toward the low-MET width shape reduces the overprediction.

For 3-prong candidates, the validation target is still negative after MC-contamination subtraction. That makes the prediction/target ratio physically uninterpretable as a closure metric. The ATLAS-like direction barely changes the prediction for either proxy (`0.995` and `1.021` relative to nominal), while the reverse stress test increases the prediction by about `17-20%`. This reinforces the earlier conclusion that the high-MET 3-prong pass-ID target is dominated by MC-contamination-subtraction/pathology rather than by a clean fake population.

Recommendation:

Do not promote width reweighting to the nominal fake estimate yet. The useful conclusion is that width/composition effects are large enough to be a credible fake-source systematic for the 1-prong high-MET transfer, and that conclusion is now supported by two stable width proxies. The next step should be to define this as a systematic-style envelope rather than a correction: compare the nominal low-MET fake estimate with the `application_to_lowmet` shifted prediction for the final 1-prong fake component. The 3-prong issue should remain a separate MC-contamination-subtraction and MC-composition problem, not be absorbed into this width-reweighting test.

Proposed systematic prescription:

- Keep the nominal fake estimate unchanged.
- Apply this uncertainty only to the **1-prong** fake component.
- Use `TauTrackWidthPt1000PV` as the representative width-composition proxy.
- Use `TauTrackWidthPt500PV` as a robustness/envelope cross-check.
- Define the shifted 1-prong fake prediction from the `application_to_lowmet` direction, because this is the ATLAS-like direction: reweight the high-MET anti-ID application region so its width shape matches the low-MET fake-factor denominator.
- Do not use `lowmet_to_application` as a nominal systematic variation. It is a stress test showing that upweighting the high-MET-like width tail worsens the agreement.
- Do not assign this width-composition uncertainty to 3-prong until the negative high-MET 3-prong `data - MC-contamination` target is understood.

In integral terms, the candidate 1-prong systematic size is:

| Width proxy | Nominal fake prediction | Shifted prediction | Relative shift |
| --- | ---: | ---: | ---: |
| `TauTrackWidthPt1000PV` | 210.924 | 167.680 | `-20.5%` |
| `TauTrackWidthPt500PV` | 210.924 | 178.643 | `-15.3%` |

### Tau-width max-weight cap validation

Question:
Does the tau-width reweighting depend on a few sparse width bins with very large shape-ratio weights?

Implementation:
- script: `run/2017/validations/validate_tau_width_reweighting.py`
- output summary: `outputs/validate_shadow_fakes/tau_width_reweighting/tau_width_reweighting_summary.md`
- CSV table: `outputs/validate_shadow_fakes/tau_width_reweighting/tau_width_max_weight_scan.csv`
- plot: `outputs/validate_shadow_fakes/tau_width_reweighting/plots/tau_width_max_weight_scan/TauTrackWidthPt1000PV_1prong_max_weight_scan.png`
- cache: `outputs/validate_shadow_fakes/tau_width_reweighting/root/validate_tau_width_max_weight_scan.root`
- scan target: `TauTrackWidthPt1000PV`, 1-prong, `application_to_lowmet` direction

Result:

| Cap | Max uncapped ratio | Capped bins | Application-shape yield fraction in capped bins | Prediction | Target | Prediction / target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uncapped | 1.267 | 0 | 0.0000 | 167.680 | 163.335 | 1.027 |
| 10 | 1.267 | 0 | 0.0000 | 167.680 | 163.335 | 1.027 |
| 5 | 1.267 | 0 | 0.0000 | 167.680 | 163.335 | 1.027 |
| 3 | 1.267 | 0 | 0.0000 | 167.680 | 163.335 | 1.027 |
| 2 | 1.267 | 0 | 0.0000 | 167.680 | 163.335 | 1.027 |

Representative appendix plot:

<img src="outputs/validate_shadow_fakes/tau_width_reweighting/plots/tau_width_max_weight_scan/TauTrackWidthPt1000PV_1prong_max_weight_scan.png" width="520">

Interpretation:

The candidate tau-width systematic does **not** require a cap for the tested production-relevant case. The largest uncapped width-shape ratio is only `1.267`, and no bins are affected by caps of `10`, `5`, `3`, or `2`. Therefore the main analysis should keep the width ratio uncapped unless a later, broader scan finds a genuinely sparse-bin instability.

For a final binned analysis implementation, the cleanest treatment is a **shape-and-normalisation uncertainty**:

```text
nominal bin = nominal 1-prong fake prediction
shifted bin = application_to_lowmet reweighted 1-prong fake prediction
uncertainty bin = max(
    |nominal - TauTrackWidthPt1000PV shifted|,
    |nominal - TauTrackWidthPt500PV shifted|
)
```

If the framework requires explicit up/down templates, this can be symmetrised around the nominal prediction:

```text
down bin = nominal bin - uncertainty bin
up bin   = nominal bin + uncertainty bin
```

with negative bins clipped only if strictly required by the downstream covariance machinery. The unsymmetrised shifted histograms should still be kept in the validation output, because they show the physical direction of the composition effect.

This should be described in the thesis as a **fake-source composition uncertainty**, not as a correction to the central fake estimate. The motivation is that the low-MET fake-factor denominator and high-MET anti-ID application region have demonstrably different tau-width shapes, and ATLAS high-mass `tau + MET` analyses use tau-width or tau-seed-width variations to assess this transfer/composition effect.

Implementation status after the 2026-06-23 cleanup:

The tau-width study is now validation-only and no longer lives in the central `analysis_shadow_unfold.py` production path. The relevant scripts are:

- `run/2017/validations/validate_tau_width_composition.py`
- `run/2017/validations/validate_tau_width_reweighting.py`

The validation implementation:

- leaves the nominal fake estimate unchanged;
- studies the 1-prong fake component, where the width-shape effect is observable;
- compares `TauTrackWidthPt1000PV` and `TauTrackWidthPt500PV`;
- tests whether an `application_to_lowmet` width reweighting moves the fake prediction in the expected direction;
- writes the diagnostic output under `outputs/validate_shadow_fakes/`.

This remains evidence for a possible **fake-source composition uncertainty**. It is not yet a nominal correction and should not be folded into the production script unless a later systematic-uncertainty implementation is explicitly chosen.

### High-MET 3-prong MC-contamination-subtraction diagnostic

The negative high-MET 3-prong validation target was then isolated with a cache-only diagnostic:

```text
run/2017/validations/validate_3prong_nonfake_subtraction.py
```

This script reads the cached low-MET fake-region validation ROOT files and does not run new dataframe event loops. The output is:

```text
outputs/validate_shadow_fakes/nonfake_subtraction/nonfake_subtraction_summary.md
```

The purpose is to separate two questions that can otherwise get conflated:

1. is the fake factor predicting the wrong number of fakes?
2. is the pass-ID `data - MC-contamination` target itself already unphysical?

The diagnostic shows that the second issue is real in the high-MET 3-prong phase space.

| Region | Prong | Data | MC contamination | Data - MC contamination | MC contamination / data |
| --- | ---: | ---: | ---: | ---: | ---: |
| `MTW >= 350`, `MET < 170` | 1 | 783.000 | 105.641 | 677.359 | 0.135 |
| `MTW >= 350`, `MET < 170` | 3 | 167.000 | 42.883 | 124.117 | 0.257 |
| `MTW >= 350`, `MET >= 170` | 1 | 1056.000 | 892.665 | 163.335 | 0.845 |
| `MTW >= 350`, `MET >= 170` | 3 | 295.000 | 353.523 | -58.523 | 1.198 |

So the signal-like high-MET 3-prong target is negative before the fake estimate is involved:

```text
295.000 data - 353.523 MC-contamination = -58.523
```

The prong-ratio comparison makes the mismatch clearer:

| Region | Quantity | 1-prong | 3-prong | 3-prong / 1-prong |
| --- | --- | ---: | ---: | ---: |
| `MTW >= 350`, `MET < 170` | data | 783.000 | 167.000 | 0.213 |
| `MTW >= 350`, `MET < 170` | total MC-contamination | 105.641 | 42.883 | 0.406 |
| `MTW >= 350`, `MET < 170` | data - MC contamination | 677.359 | 124.117 | 0.183 |
| `MTW >= 350`, `MET >= 170` | data | 1056.000 | 295.000 | 0.279 |
| `MTW >= 350`, `MET >= 170` | total MC-contamination | 892.665 | 353.523 | 0.396 |
| `MTW >= 350`, `MET >= 170` | data - MC contamination | 163.335 | -58.523 | -0.358 |

The high-MET MC-contamination remains much more 3-prong-heavy than data. The component breakdown shows that this is dominated by `wtaunu_had`:

| Region | Prong | Component | Yield | Fraction of MC contamination | Fraction of data |
| --- | ---: | --- | ---: | ---: | ---: |
| `MTW >= 350`, `MET >= 170` | 1 | `wtaunu_had` | 710.987 | 0.796 | 0.673 |
| `MTW >= 350`, `MET >= 170` | 3 | `wtaunu_had` | 289.671 | 0.819 | 0.982 |
| `MTW >= 350`, `MET >= 170` | 3 | all other MC-contamination | 63.852 | 0.181 | 0.216 |

In other words, in the signal-like high-MET 3-prong ID region, `wtaunu_had` alone is almost as large as the observed data. Once the other MC-contamination components are added, the total MC-contamination prediction overshoots data by about `58.5` events.

The bin-by-bin residual is not a single-bin accident. Most high-MET 3-prong `MTW` bins have MC-contamination at or above data:

| MTW bin [GeV] | Data | MC contamination | Data - MC contamination | MC contamination / data |
| --- | ---: | ---: | ---: | ---: |
| 350-375 | 36.000 | 55.400 | -19.400 | 1.539 |
| 375-400 | 47.000 | 56.091 | -9.091 | 1.193 |
| 400-430 | 50.000 | 57.889 | -7.889 | 1.158 |
| 430-465 | 52.000 | 51.763 | 0.237 | 0.995 |
| 465-500 | 28.000 | 34.256 | -6.256 | 1.223 |
| 500-550 | 24.000 | 30.645 | -6.645 | 1.277 |
| 550-600 | 23.000 | 21.459 | 1.541 | 0.933 |
| 600-700 | 19.000 | 22.321 | -3.321 | 1.175 |
| 700-850 | 12.000 | 13.321 | -1.321 | 1.110 |
| 850-1000 | 4.000 | 5.523 | -1.523 | 1.381 |
| 1000-2000 | 0.000 | 4.856 | -4.856 | undefined |

This changes how the high-MET 3-prong proxy should be used. It is **not** a clean fake-factor validation target, because the target being predicted is already negative. It is better treated as a MC-contamination-subtraction diagnostic. The fake-factor validation should rely more heavily on the control-MET target, where both prongs have positive pass-ID residuals and the low-MET fake-enriched derivation improves the prediction.

Recommended interpretation:

- The low-MET fake-enriched method remains promising because it improves the independent `MTW >= 350`, `MET < 170` validation.
- The high-MET 3-prong failure should not be used as evidence that the low-MET fake-factor method is wrong.
- The remaining high-MET 3-prong issue points back to reconstructed `wtaunu_had` MC-contamination modelling, especially the prong-dependent ID-region yield after weighting.
- Any thesis correction should explicitly separate fake-factor transfer validation from this MC-contamination-subtraction pathology.

### High-MET threshold prong-balance extension

Question:
The previous diagnostic showed that the high-MET, nominal-MTW 3-prong pass-ID target is negative after MC-contamination subtraction. The next check asked whether that problem appears as soon as `MET >= 170`, or whether it is mainly tied to the nominal `MTW >= 350` signal-like corner.

Implementation:

- script: `run/2017/validations/validate_prong_balance_thresholds.py`
- output summary: `outputs/validate_shadow_fakes/prong_balance_thresholds/prong_balance_thresholds_summary.md`
- corrected ROOT cache: `outputs/validate_shadow_fakes/prong_balance_thresholds/root/validate_prong_balance_thresholds_mtw300.root`
- mode: validation-only, nominal only, `MTW` only, medium tau ID, split 1-prong and 3-prong, no unfolding
- technical detail: the validation uses an `MTW` binning with a leading `300-350 GeV` bin. The standard thesis binning starts at `350 GeV`, so it would otherwise throw away the exact shadow interval being tested.

The test compares the current MTW-shadow fake-factor derivation against the low-MET fake-enriched derivation. For each region it computes the `wtaunu_had` scale factor that would be implied after subtracting the fake estimate and all other MC-contamination:

```text
implied wtaunu_had = data - fakes - other MC-contamination
implied SF = implied wtaunu_had / nominal wtaunu_had MC
```

Result:

| Region | Fake model | SF 1-prong | SF 3-prong | SF 3-prong / SF 1-prong |
| --- | --- | ---: | ---: | ---: |
| `300 <= MTW < 350`, `MET >= 170` | current MTW-shadow CR | 0.777 | 0.735 | 0.946 |
| `300 <= MTW < 350`, `MET >= 170` | low-MET fake-enriched CR | 0.767 | 0.692 | 0.902 |
| `MTW >= 300`, `MET >= 170` | current MTW-shadow CR | 0.935 | 0.750 | 0.803 |
| `MTW >= 300`, `MET >= 170` | low-MET fake-enriched CR | 0.925 | 0.743 | 0.803 |

The detailed yields are:

| Region | Fake model | Prong | Data | Fakes | Other MC contamination | `wtaunu_had` MC | Data - fakes - other MC contamination |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `300 <= MTW < 350`, `MET >= 170` | current MTW-shadow CR | 1 | 52.000 | 13.209 | 10.840 | 35.962 | 27.951 |
| `300 <= MTW < 350`, `MET >= 170` | current MTW-shadow CR | 3 | 12.000 | 0.574 | 2.422 | 12.252 | 9.004 |
| `300 <= MTW < 350`, `MET >= 170` | low-MET fake-enriched CR | 1 | 52.000 | 13.569 | 10.840 | 35.962 | 27.592 |
| `300 <= MTW < 350`, `MET >= 170` | low-MET fake-enriched CR | 3 | 12.000 | 1.097 | 2.422 | 12.252 | 8.480 |
| `MTW >= 300`, `MET >= 170` | current MTW-shadow CR | 1 | 1108.000 | 217.007 | 192.518 | 746.949 | 698.476 |
| `MTW >= 300`, `MET >= 170` | current MTW-shadow CR | 3 | 307.000 | 14.141 | 66.275 | 301.923 | 226.585 |
| `MTW >= 300`, `MET >= 170` | low-MET fake-enriched CR | 1 | 1108.000 | 224.493 | 192.518 | 746.949 | 690.989 |
| `MTW >= 300`, `MET >= 170` | low-MET fake-enriched CR | 3 | 307.000 | 16.309 | 66.275 | 301.923 | 224.416 |

Interpretation:
The narrow `300 <= MTW < 350`, `MET >= 170` region does not show the same strong prong-specific imbalance as the full high-MET region. Both prongs prefer a lower `wtaunu_had` normalisation there, but the relative 3-prong suppression is mild: `SF_3p / SF_1p` is about `0.90-0.95`.

The broader `MTW >= 300`, `MET >= 170` region already shows the stronger pattern seen in the nominal signal-like region: the 3-prong implied scale is about `20%` lower than the 1-prong implied scale. This means the issue is not simply "any high MET" and not just the first `300-350 GeV` shadow interval. The problematic behaviour enters once the selection includes the higher-MTW tail, where `wtaunu_had` becomes dominant and the reconstructed-weighted 3-prong component is too large relative to data.

Recommendation:
Do not introduce a global 3-prong scale factor from this check. The needed correction is region-dependent: the `300-350` high-MET interval would imply only a mild relative 3-prong adjustment, while `MTW >= 300` implies a stronger one. This supports keeping the current evidence as a modelling diagnostic and focusing next on either:

1. a high-MTW/topology-dependent MC-contamination modelling uncertainty for `wtaunu_had`, or
2. a fake-model alternative such as the low-MET fake-enriched derivation, tested in the full unfolding workflow as a validation/systematic rather than as an immediate central-value replacement.

### ATLAS precedent for a more differential fake factor

No exact ATLAS precedent was found for this specific prescription, i.e. a literal two-dimensional `(TauPt, MET_met)` fake factor in a charged-current `tau + MET` unfolding measurement. The defensible statement is narrower: recent ATLAS fake-background work supports using fake factors that depend on more than one piece of event or tau-candidate information when validation shows that a one-dimensional transfer is not stable.

The strongest reference is the ATLAS Universal Fake Factor paper:

- ATLAS Collaboration, "Estimation of backgrounds from jets misidentified as tau-leptons using the Universal Fake Factor method with the ATLAS detector", arXiv:2502.04156, EPJC 85 (2025) 1441: https://arxiv.org/abs/2502.04156

That paper motivates data-driven fake factors because jets misidentified as hadronic tau decays are not reliably modelled in simulation. It also states that fake-factor uncertainties depend on the tau-lepton transverse momentum and charged-particle decay multiplicity, and that the fake rate depends on the underlying fake-source composition. The UFF method addresses this by combining fake factors measured in samples enriched in light-quark, gluon, b-quark, and pile-up fake sources.

This is directly relevant to the present validation result. The current prong-split `TauPt` fake factor transfers well across MTW sidebands but poorly across MET sidebands, especially for 3-prong candidates. Since the final signal region is high-MET, this suggests that the fake-source composition or fake-rate behaviour changes with `MET_met`. Testing a prong-split `(TauPt, MET_met)` fake factor is therefore an analysis-specific extension of the same ATLAS logic: keep the established tau-kinematic and prong dependence, and add the variable where the transfer failure is observed.

Two additional ATLAS references give useful context:

- ATLAS Collaboration, "Tools for estimating fake/non-prompt lepton backgrounds with the ATLAS detector at the LHC", arXiv:2211.16178: https://arxiv.org/abs/2211.16178
- ATLAS Collaboration, "Search for high-mass resonances in final states with a tau-lepton and missing transverse momentum with the ATLAS detector", arXiv:2402.16576: https://arxiv.org/abs/2402.16576

The first is a general ATLAS fake-background methods reference. It states that fake/non-prompt efficiencies can depend on momentum, proximity to other objects, and other analysis-dependent factors, and that fake-factor methods use the fake factor appropriate to each event. The second is the closest topology reference: hadronic tau plus missing transverse momentum, with transverse mass as the final discriminant and jet-fake backgrounds estimated from data. It supports the relevance of validating fake modelling in `tau + MET` phase space, although it should not be cited as proof that ATLAS used this exact 2D fake-factor form.

The recommended wording for the thesis is therefore:

> ATLAS fake-factor methods show that jet-to-tau fake rates depend on tau kinematics, charged-particle multiplicity, and fake-source composition. Since the validation here shows poor transfer across `MET_met` sidebands, a prong-split fake factor differential in both `TauPt` and `MET_met` is a natural analysis-specific extension to test. This is not adopted as a nominal correction until validated in independent sidebands.

## Changes Relative To The Thesis-Version Analysis

The thesis image snapshot corresponds to the older analysis state. The current analysis differs in several material ways.

### 1. Fake-background implementation corrected

The fake estimate implementation was corrected to match the intended fake-factor method.

The material change is commit `5d9535a` (`Fix fake factor implementation`). That
commit fixed two mistakes in `Analysis.do_fakes_estimate()`:

1. the MC subtraction used in the fake-factor numerator and denominator had the
   pass-ID and fail-ID selections swapped;
2. the target-variable fake prediction was built from fake-factor-weighted
   ID-region MC rather than from fake-factor-weighted anti-ID data with MC-contamination
   MC subtracted.

Before the fix, the effective fake estimate was therefore not the intended
data-driven anti-ID transfer. The corrected target-variable estimate is:

```text
fake prediction in target variable =
    FF(source variable) * SR_failID_data
  - FF(source variable) * trueTau_SR_failID_MC
```

where:

```text
FF = (CR_passID_data - trueTau_CR_passID_MC)
   / (CR_failID_data - trueTau_CR_failID_MC)
```

The important conceptual change is that the target-variable fake shape is built
from the fail-ID side of the method, not from weighted pass-ID MC. This matters
because the fake estimate should be data-driven in the fail-ID control side and
then transferred into the pass-ID signal side using the fake factor.

Expected consequence:

- fake-background shapes and normalisations can change relative to the thesis plots;
- unfolded data can move because the background-subtracted measured input changes.

### 2. `W -> tau nu -> hadrons` and `W -> tau nu -> leptons` are separated

The current samples split the old `wtaunu` treatment into:

- `wtaunu_had`: signal, with `TruthTau_isHadronic`;
- `wtaunu_lep`: background, corresponding to leptonic tau decays.

Expected consequence:

- plot paths and legends now refer to `wtaunu_had` and `wtaunu_lep`;
- current systematic folders differ from thesis folders that used `wtaunu`;
- the signal definition is cleaner for a hadronic-tau fiducial measurement.

### 3. Shadow-bin unfolding is now tested with matched measured inputs and responses

The thesis-style shadow treatment used a post-unfolding shadow-bin acceptance scaling. The first full-shadow test changed the response matrix but still used nominal measured inputs, which did not give a physical closure test.

The current `analysis_shadow_unfold.py` workflow regenerates the measured inputs, backgrounds, fakes, and response in the same variable-specific shadow phase space.

Expected consequence:

- the current closure test is not directly comparable to the thesis plots as a presentation-only change;
- it is testing a different and more internally consistent unfolding construction.

### 4. Nonfiducial signal is subtracted before unfolding

This is the most important closure change.

The current analysis subtracts reconstructed `wtaunu_had` events that pass the reco selection but fail the nominal truth fiducial definition before unfolding. The response reco projection is also built from truth-fiducial reconstructed signal.

Expected consequence:

- signal-MC closure improves from order-10% biases to per-mille agreement;
- the unfolded data result is not forced to match MC, because only signal outside the fiducial target is removed.

### 5. Temporary MTW/MET category diagnostic was removed

A diagnostic `MTW_METCategoryMTW` variable was tested to distinguish MTW-shadow, MET-shadow, and combined shadow reconstructed regions. After the nonfiducial correction, this was no longer needed for the current workflow and has been removed from `analysis_shadow_unfold.py`.

Expected consequence:

- future reruns should no longer produce `MTW_MET_category_shadow_bin_250` plots;
- any existing category plots in `outputs/analysis_shadow_unfold` are historical diagnostics only.

### 6. Current shadow-closure output is central-value only

The current closure test has:

`DO_FULL_SYSTEMATICS = False`

This is deliberate. It establishes central-value signal-MC closure first.

Expected consequence:

- the current report does not claim a final uncertainty band;
- full systematics still need to be enabled and checked before this can replace the final thesis unfolding result.

### 7. Presentation and tooling changes

Several changes are not physics changes but affect generated outputs:

- ATLAS label default is now blank rather than printing `Internal`.
- ROOT and `mplhep` compatibility updates were made.
- histogram-production scope was narrowed in some scripts to avoid unnecessary ROOT event-loop cost.
- old `run/2017_viva` files are treated as historical backup and excluded from active checking.

These should not be interpreted as physics differences.

### 8. Production script and validation scripts are now separated

The current `analysis_shadow_unfold.py` script has been cleaned up so that it is no longer the validation playground. It now owns the central production-style closure workflow:

- variable: `MTW`;
- configurations: no-shadow plus `MTW` shadow thresholds `200`, `250`, and `300` GeV;
- tau ID working point: `medium`;
- fake estimate: prong-split `TauPt` fake factors;
- fake derivation region: low-MET fake-enriched region with `TauPt > 170` and `MET_met < 100`;
- nonfiducial signal subtraction before unfolding;
- central-value response/unfolding with `DO_FULL_SYSTEMATICS = False` by default.

Validation and diagnostic logic that used to live inside `analysis_shadow_unfold.py` has been moved into dedicated scripts under:

```text
run/2017/validations/
```

The moved validations include:

| Validation question | Script |
|---|---|
| fake-scale unfolding impact | `run/2017/validations/validate_fake_scale_unfolding.py` |
| MC-only fake closure | `run/2017/validations/validate_mc_fake_closure.py` |
| split-sample MC unfolding closure | `run/2017/validations/validate_split_sample_unfolding_closure.py` |
| propagated `wtaunu_had` prong-composition impact | `run/2017/validations/validate_prong_model_unfolding_impact.py` |
| inclusive versus prong-split fake factors | `run/2017/validations/validate_inclusive_prong_fakes.py` |
| tau-width fake-source composition/reweighting | `run/2017/validations/validate_tau_width_composition.py`, `run/2017/validations/validate_tau_width_reweighting.py` |
| low-MET and ATLAS-like fake-region tests | `run/2017/validations/validate_low_met_fake_region.py`, `run/2017/validations/validate_atlas_like_fake_transfer.py` |

The physics status of these tests is unchanged: they remain validation evidence, not nominal corrections. The important implementation change is organisational. Future thesis write-up should describe `analysis_shadow_unfold.py` as the central MTW shadow-unfolding workflow and cite the validation scripts separately when discussing alternative fake models or modelling uncertainties.

### 9. Fake-factor runtime optimisation

The low-MET fake-enriched control region is identical for all shadow-bin configurations:

```text
passReco, baseline tau, passMetTrigger, eta acceptance, TauPt > 170, MET_met < 100
```

Previously, `analysis_shadow_unfold.py` recomputed the fake-factor numerator, denominator, and source-variable fake factor separately for every shadow-bin configuration and prong. The 2026-06-23 run showed that this was expensive: each inclusive, 1-prong, or 3-prong fake estimate took roughly `8-9` minutes, and the same low-MET fake-factor derivation was repeated for each configuration.

The script now marks the active low-MET fake control region as shared:

```python
shared_across_configs=True
```

For shared control regions, the script computes each nominal fake factor once per prong category and fake-factor source binning:

- 1-prong;
- 3-prong.

Later shadow-bin configurations reuse the cached fake-factor histogram and only run the SR fail-ID application step for the current target selection. The cache key includes the `TauPt` fake-factor bin edges, so configurations with different `TauPt` shadow-bin source binnings do not accidentally reuse an incompatible fake factor. This does not change the fake-factor algebra or the nominal physics definition. It only avoids rescanning the same low-MET CR numerator and denominator repeatedly when the CR and source binning are genuinely identical.

The first-time fake-factor build has also been consolidated into
`src/fakes.py::build_fake_factor_batched`. The public
`Analysis.do_fakes_estimate()` method now delegates to this implementation, so
the framework no longer carries two separate fake-factor algorithms. The helper
books the required `TauPt` CR/SR source histograms before collecting results.
This keeps the same fake-factor equation but avoids the old `get_hist(...,
allow_generation=True)` pattern where each missing source histogram could
trigger its own ROOT event loop. The SR fail-ID target-variable application is
still handled separately because it depends on the current target selection.

The original thesis-style CR is commented out with `shared_across_configs=False`, because that CR depends on the shadow-bin thresholds and should not reuse fake factors across configurations without an explicit validation.

Inclusive fake factors are no longer part of the main production script. They remain a validation-only comparison in `run/2017/validations/validate_inclusive_prong_fakes.py`.

### 10. Central low-MET fake-model rerun

Question:
Does the current central `analysis_shadow_unfold.py` configuration give a coherent
MTW unfolding after the bookkeeping cleanup?

Implementation:
- script: `run/2017/analysis_shadow_unfold.py`
- mode: cached central-value rerun, no full systematics
- log: `outputs/analysis_shadow_unfold/logs/analysis_shadow_unfold_2026-06-24_09-12-13.log`
- summary: `outputs/analysis_shadow_unfold/closure_summary.md`
- active fake-factor determination region: `TauPt > 170`, `MET_met < 100`
- active fake model: prong-split `TauPt` fake factor
- active bookkeeping: MC-contamination subtraction plus data-driven jet-fake estimate
- active fake-source variation: 1-prong `TauTrackWidthPt1000PV` application-to-lowMET shape shift

Result:

| Configuration | Signal-MC closure integral | Data | MC-contam bkg | Jet-fake-like MC bkg | Fakes | Nonfid signal | Nominal data sig | Fid reco / nominal data sig | Width-shifted data sig | Fid reco / width-shifted data sig |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_shadow_bin | 1.000 | 1351.000 | 245.530 | 113.607 | 226.136 | 104.534 | 774.800 | 1.157 | 818.044 | 1.096 |
| MTW_shadow_bin_250 | 1.000 | 1428.000 | 264.066 | 124.315 | 243.815 | 120.776 | 799.343 | 1.169 | 844.923 | 1.106 |

Representative outputs:
- nominal no-shadow unfolded MTW:
  `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/no_shadow_bin_MTW_2iter_unfolded.png`
- nominal MTW shadow-bin 250 unfolded MTW:
  `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/MTW_shadow_bin_250_MTW_2iter_unfolded.png`
- no-shadow tau-width shift:
  `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/fake_width_systematic/no_shadow_bin_MTW_TauTrackWidthPt1000PV_2iter_fake_width_shift.png`
- MTW shadow-bin 250 tau-width shift:
  `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/fake_width_systematic/MTW_shadow_bin_250_MTW_TauTrackWidthPt1000PV_2iter_fake_width_shift.png`

Interpretation:
The central signal-MC self-closure remains exact at the integral level and at
the per-mille/bin-percent level for the iterative checks. The remaining issue
is therefore not the response machinery. It is the measured input normalisation.

The MC-contamination bookkeeping correction has kept the result in a much more
reasonable range than the old all-MC-background convention, but the nominal
data input is still low relative to the fiducial reconstructed signal by about
`16-17%`. The validated 1-prong tau-width fake-source shift reduces that
deficit to about `10%`. Visually, the shifted unfolded data points move toward
the truth/signal-MC curve, especially in the populated low-to-mid MTW bins, but
this should still be described as a fake-source composition variation rather
than a final central correction.

Recommendation:
This run is good enough to use as the current central comparison point. The
next expensive step should not be another fake-validation sweep. The next useful
production step is a full-systematics rerun only if we accept the current
central prescription as the working model. Otherwise, the remaining analysis
decision is whether the tau-width shift becomes a systematic envelope, a central
fake-model correction, or remains validation-only.

## Implemented High-Priority Fake-Source Systematics

Question:
Which fake-source uncertainties are now implemented in the cleaned
`analysis_shadow_unfold.py` workflow?

Implementation:
- script: `run/2017/analysis_shadow_unfold.py`
- module: `run/2017/shadow_unfold/systematics.py`
- mode: production analysis support; no new result has been produced from this
  code change yet
- DTA input data: read-only; no writes are made under `/mnt/D/data/DTA_outputs`

The shadow-unfold workflow now has three fake-source systematic switches:

| Switch | Systematic name | Meaning |
|---|---|---|
| `RUN_FAKE_FF_STAT_SYSTEMATIC` | `JET_FAKE_FF_STAT` | shifts each fake-factor bin by its statistical uncertainty and unfolds the shifted inputs |
| `RUN_FAKE_MET_WINDOW_SYSTEMATIC` | `JET_FAKE_MET_WINDOW` | envelopes alternate low-MET fake-factor regions `[30,100]`, `[50,100]`, `[70,100]`, and `[0,150]` GeV against the nominal `[0,100]` GeV region |
| `RUN_FAKE_WIDTH_SYSTEMATIC` | `JET_FAKE_TAU_WIDTH_COMPOSITION` | applies the validated 1-prong `TauTrackWidthPt1000PV` fake-source composition shift as an uncertainty envelope |

Representative outputs expected after the next completed run:
- closure/systematic table:
  `outputs/analysis_shadow_unfold/closure_summary.md`
- fake-factor statistical uncertainty plots and supporting checks:
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_source_systematics/*fake_factor_stat_uncertainty.png`
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_source_systematics/*fake_factor_stat_source.png`
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_source_systematics/*fake_factor_stat_fake_yield.png`
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_source_systematics/*fake_factor_stat_data_sig.png`
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_source_systematics/*fake_factor_stat_unfolded_shift.png`
- MET-window uncertainty plots and supporting checks:
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_source_systematics/*met_window_uncertainty.png`
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_source_systematics/*met_window_fake_factors.png`
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_source_systematics/*met_window_fake_yield.png`
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_source_systematics/*met_window_data_sig.png`
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_source_systematics/*met_window_unfolded_shift.png`
- tau-width uncertainty plots and supporting checks:
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_width_systematic/*fake_width_uncertainty.png`
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_width_systematic/*transfer_weight.png`
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_width_systematic/*fake_yield.png`
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_width_systematic/*data_sig.png`
- combined fake-source uncertainty:
  `outputs/analysis_shadow_unfold/plots/<config>/MTW/fake_source_systematics/*combined_fake_source_uncertainty.png`

Interpretation:
These are uncertainty envelopes, not central fake-model changes. The nominal
unfolded data still uses the central prong-split, low-MET, `TauPt`-sourced fake
estimate. The shifted fake predictions are unfolded with the same nominal
response and compared bin-by-bin to the nominal unfolded result.

The supporting plots are intended to verify the direction of the uncertainty
propagation. A shifted fake factor should first change the predicted fake yield,
then shift the background-subtracted `data_sig` in the opposite direction, and
only then move the unfolded spectrum. This mirrors the thesis prescription of
checking the source variation before interpreting the final percentage
uncertainty.

Recommendation:
Run `analysis_shadow_unfold.py` once with the three fake-source systematic
switches enabled and `DO_FULL_SYSTEMATICS = False`. If the fake-source
uncertainties are numerically stable, the next extension is the separate
detector/response systematic run controlled by `DO_FULL_SYSTEMATICS`.

## Completed Fake-Source Systematics Rerun

Question:
After implementing the high-priority fake-source systematics, does the current
central shadow-unfold result behave coherently, and which uncertainty dominates?

Implementation:
- script: `run/2017/analysis_shadow_unfold.py`
- mode: production-style central run with fake-source systematic envelopes
- log: `outputs/analysis_shadow_unfold/logs/analysis_shadow_unfold_2026-06-24_10-33-32.log`
- summary: `outputs/analysis_shadow_unfold/closure_summary.md`
- active fake-factor derivation region: `TauPt > 170`, `MET_met < 100`
- active fake model: prong-split `TauPt` fake factor
- active systematics:
  - `JET_FAKE_FF_STAT`
  - `JET_FAKE_MET_WINDOW`
  - `JET_FAKE_TAU_WIDTH_COMPOSITION`
- response systematics: not enabled in this run (`DO_FULL_SYSTEMATICS = False`)

Central pre-unfolding budget:

| Configuration | Data | All MC bkg | MC-contam bkg | Jet-fake-like MC bkg | Fakes | Nonfid signal | Nominal data sig | Old all-MC diagnostic data sig | Data sig, no fakes | Fid reco / data sig |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_shadow_bin | 1351.000 | 359.137 | 245.530 | 113.607 | 226.136 | 104.534 | 774.800 | 661.192 | 1000.936 | 1.157 |
| MTW_shadow_bin_250 | 1428.000 | 388.381 | 264.066 | 124.315 | 243.815 | 120.776 | 799.343 | 675.028 | 1043.158 | 1.169 |

Fake-source uncertainty summary:

| Configuration | Systematic | Iteration | Fake-yield change | Data-sig change | Unfolded relative integral shift |
|---|---|---:|---:|---:|---:|
| no_shadow_bin | `JET_FAKE_FF_STAT` | 2 | `226.136 -> 228.357 / 223.915` | `774.800 -> 772.579 / 777.021` | 0.3% |
| no_shadow_bin | `JET_FAKE_MET_WINDOW` | 2 | `226.136 -> 226.498 / 216.482` | `774.800 -> 774.438 / 784.454` | 1.2% |
| no_shadow_bin | `JET_FAKE_TAU_WIDTH_COMPOSITION` | 2 | `226.136 -> 182.892` | `774.800 -> 818.044` | 6.6% |
| MTW_shadow_bin_250 | `JET_FAKE_FF_STAT` | 2 | `243.815 -> 246.102 / 241.528` | `799.343 -> 797.056 / 801.630` | 0.3% |
| MTW_shadow_bin_250 | `JET_FAKE_MET_WINDOW` | 2 | `243.815 -> 244.198 / 233.721` | `799.343 -> 798.960 / 809.437` | 1.2% |
| MTW_shadow_bin_250 | `JET_FAKE_TAU_WIDTH_COMPOSITION` | 2 | `243.815 -> 198.235` | `799.343 -> 844.923` | 6.5% |

Representative outputs:
- nominal no-shadow unfolded MTW:
  `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/no_shadow_bin_MTW_2iter_unfolded.png`
- nominal MTW shadow-bin 250 unfolded MTW:
  `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/MTW_shadow_bin_250_MTW_2iter_unfolded.png`
- combined no-shadow fake-source uncertainty:
  `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/fake_source_systematics/no_shadow_bin_MTW_2iter_combined_fake_source_uncertainty.png`
- combined MTW shadow-bin 250 fake-source uncertainty:
  `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/fake_source_systematics/MTW_shadow_bin_250_MTW_2iter_combined_fake_source_uncertainty.png`
- no-shadow tau-width shifted unfolded spectrum:
  `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/fake_width_systematic/no_shadow_bin_MTW_TauTrackWidthPt1000PV_2iter_fake_width_shift.png`
- MTW shadow-bin 250 tau-width shifted unfolded spectrum:
  `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/fake_width_systematic/MTW_shadow_bin_250_MTW_TauTrackWidthPt1000PV_2iter_fake_width_shift.png`

Interpretation:
The same-sample signal-MC closure remains essentially exact: the closure summary
gives integral ratio `1.000` for both configurations and all tested iterations,
with per-bin deviations at the per-mille to percent level. The response and
RooUnfold bookkeeping therefore look internally consistent.

The measured data input remains low relative to the fiducial reconstructed
signal. With the current MC-contamination plus data-driven fake bookkeeping,
`Fid reco / data sig` is `1.157` for no-shadow MTW and `1.169` for the
MTW-shadow-250 configuration. The old all-MC-background diagnostic is worse
(`661.192` and `675.028` data-sig integrals), so the bookkeeping correction is
still the right direction.

The fake-factor statistical uncertainty is negligible at the analysis scale.
The MET-window envelope is visible but modest. The 1-prong tau-width
composition envelope is the dominant fake-source systematic: it reduces the
predicted fake yield by roughly `43-46` events, raises the background-subtracted
data input, and moves the unfolded data toward the signal-MC/truth spectrum.
Even after that shift, the fiducial reco/data-sig ratio remains above unity
(`1.096` and `1.106`), so the tau-width effect improves but does not fully solve
the data/MC normalisation difference.

Recommendation:
Use this run as the current central fake-source systematic reference. The next
physics-producing run should be a detector/response systematic pass only if the
current central prescription is accepted as the working model. Before that, the
open analysis judgement is whether `JET_FAKE_TAU_WIDTH_COMPOSITION` should stay
as a systematic envelope, or whether more evidence is needed before it can be
promoted to a central fake-model correction. The current evidence supports it
as an uncertainty envelope, not a central correction.

## Full-Systematics Smoke Run

Question:
When `DO_FULL_SYSTEMATICS = True`, do the response-systematic plots contain the
full set of response variations needed for a final uncertainty band?

Implementation:
- script: `run/2017/analysis_shadow_unfold.py`
- mode: full-systematics smoke run
- log: `outputs/analysis_shadow_unfold/logs/analysis_shadow_unfold_2026-06-24_14-22-21.log`
- summary: `outputs/analysis_shadow_unfold/closure_summary.md`
- response plots:
  - `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/systematics/no_shadow_bin_MTW_response_systematics.png`
  - `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/systematics/MTW_shadow_bin_250_MTW_response_systematics.png`

Result:
The `2026-06-24 14:22` run completed and produced fresh central unfolding,
fake-source systematic, response-matrix, covariance, and response-systematic
plots for both active configurations. The response cache guard also worked: it
detected stale cached truth-selection cutflows and rebuilt the response cache
before importing it.

However, the response-systematics result is still not complete. The script
skipped the same eight TES response variations for both configurations:

| Skipped response systematic group |
| --- |
| `TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_HighPt` |
| `TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_LowPt` |
| `TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_HighPt` |
| `TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt` |
| `TAUS_TRUEHADTAU_SME_TES_INSITUEXP` |
| `TAUS_TRUEHADTAU_SME_TES_INSITUFIT` |
| `TAUS_TRUEHADTAU_SME_TES_MODEL_CLOSURE` |
| `TAUS_TRUEHADTAU_SME_TES_PHYSICSLIST` |

The latest logs show the reason explicitly: the TES shifted trees emit warnings
that the truth columns needed for `TruthMTW` are unavailable, and the varied
response objects are then empty or non-finite. The current response-systematic
plots therefore do not represent a complete detector/response uncertainty band.
They should be treated as a diagnostic of the plotting path only.

The visual form of the current response-systematic plot is also not thesis
ready: the included curves sit near `100%` on an axis labelled "Response
uncertainty / %". That is not a physically interpretable final uncertainty
presentation. Before quoting this figure, the plotting convention needs to be
checked so it shows an actual fractional deviation from nominal, not a
nominal-normalised ratio plotted as a percent.

Fake-source systematics in the same run are usable and remain consistent with
the previous central run. With the 4-iteration setting enabled, the dominant
fake-source contribution is still the tau-width composition envelope. The
combined fake-source uncertainty reaches roughly `20-25%` in the lowest `MTW`
bin and about `10-12%` in the sparse high-`MTW` tail. The labels in the
diagnostic plots are too long for thesis use and should be cleaned before final
figure export.

Representative fake-source plots:
- `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/fake_source_systematics/no_shadow_bin_MTW_4iter_combined_fake_source_uncertainty.png`
- `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/fake_source_systematics/MTW_shadow_bin_250_MTW_4iter_combined_fake_source_uncertainty.png`

Interpretation:
This is a successful smoke run, not a final full-systematics result. It proves
that the response-systematic plotting path works for complete response
variations, and it protects us from silently including incomplete TES response
objects. It also shows that the current response uncertainty plot is missing the
TES component, which is too important to omit without a dedicated justification.

Recommendation:
Do not quote the current response-systematic plot as the final detector/response
uncertainty. The next response-systematics task is to decide how TES variations
should be propagated through the response:

- regenerate or book TES `truth_reco_tau` response histograms if the shifted
  trees can support the truth+reco selection; or
- implement a defensible reco-only TES response treatment if the shifted trees
  cannot provide matched truth+reco objects; or
- explicitly exclude TES from this response plot only after documenting why it
  is handled elsewhere.

### Response-systematic Failure Diagnosis

Question:
Why are the TES response systematics skipped, and why do the currently included
response-systematic curves sit near `100%`?

Implementation:
- script inspected: `run/2017/analysis_shadow_unfold.py`
- helper inspected: `run/2017/shadow_unfold/systematics.py`
- diagnostic script: `run/2017/validations/validate_response_weight_normalisation.py`
- TES truth-axis diagnostic: `run/2017/validations/validate_tes_response_truth_axis.py`
- ROOT cache inspected: `outputs/analysis_shadow_unfold/response/root/wtaunu_had.root`
- diagnostic outputs:
  - `outputs/validate_shadow_fakes/response_weight_normalisation/response_weight_normalisation_summary.md`
  - `outputs/validate_shadow_fakes/response_weight_normalisation/response_weight_normalisation.csv`
  - fresh response cache: `outputs/validate_shadow_fakes/response_weight_normalisation/fresh_standard_response/root/wtaunu_had.root`
  - `outputs/validate_shadow_fakes/tes_response_truth_axis/tes_response_truth_axis_summary.md`
  - `outputs/validate_shadow_fakes/tes_response_truth_axis/tes_response_truth_axis_tree_checks.csv`
- logs inspected:
  - `outputs/analysis_shadow_unfold/logs/analysis_shadow_unfold_2026-06-24_14-22-21.log`
  - `outputs/analysis_shadow_unfold/response/logs/analysis_shadow_unfold_response_build_2026-06-24_14-22-52.log`
  - `outputs/analysis_shadow_unfold/response/logs/build_shadow_response_tes_no_shadow_bin_2026-06-24_14-24-25.log`
  - `outputs/analysis_shadow_unfold/response/logs/build_shadow_response_tes_MTW_shadow_bin_250_2026-06-24_14-25-05.log`

Result:

| Check | Result | Interpretation |
|---|---:|---|
| TES shifted `MTW` reco histogram, no-shadow | `898.854` | reco projection exists and is finite |
| TES shifted `MTW_TruthMTW` matrix, no-shadow | `0.000` | full TES response matrix is absent/empty |
| TES shifted `MTW` reco histogram, `MTW_shadow_bin_250` | `935.888` | reco projection exists and is finite |
| TES shifted `MTW_TruthMTW` matrix, `MTW_shadow_bin_250` | `0.000` | full TES response matrix is absent/empty |
| Nominal no-shadow `truth_reco_tau` `MTW` | `896.196` | central response normalization |
| `TAUS_TRUEHADTAU_EFF_TRIGGER_STATDATA161718__1up` no-shadow `truth_reco_tau` `MTW` | `59420.415` | `66.3x` nominal; not a physical response uncertainty |
| Nominal no-shadow reco-only `MTW` | `1000.730` | reco-only control selection |
| `TAUS_TRUEHADTAU_EFF_TRIGGER_STATDATA161718__1up` reco-only `MTW` | `1061.025` | `1.06x` nominal; physically plausible |

The TES truth-axis diagnostic then inspected the shifted input trees and current
response cache directly:

| TES diagnostic check | Result | Interpretation |
|---|---:|---|
| Representative TES missing truth inputs | `VisTruthTauPt`, `VisTruthTauPhi`, `TruthNeutrinoPt`, `TruthNeutrinoPhi`, `VisTruthTauEta`, `TruthTau_nChargedTracks` | shifted TES trees do not carry the inputs needed for `TruthMTW` |
| Representative TES missing reco inputs | none | shifted TES trees do carry the reconstructed inputs needed for shifted `MTW` |
| TES entries matched to nominal event keys | `1.000-1.000` | nominal truth lookup by `(mcChannel,eventNumber)` is feasible |
| TES response reco histograms with non-zero integral | `32` | shifted reco projections exist in the current cache |
| TES response matrices with non-zero integral | `0` | full shifted migration matrices are not currently built |

The targeted diagnostic was then rerun with a fresh standard response cache
after fixing the systematic histogram booking in `src/dataset.py`. The important
comparison is:

| Source | Selection | EFF variations checked | Incompatible variations | Min varied/nominal | Max varied/nominal |
|---|---|---:|---:|---:|---:|
| current main response cache | `no_shadow_bin_medium_truth_reco_tau` | 30 | 30 | `58.68` | `66.30` |
| current main response cache | `MTW_shadow_bin_250_medium_truth_reco_tau` | 30 | 30 | `57.31` | `64.77` |
| fresh standard response cache after fix | `no_shadow_bin_medium_truth_reco_tau` | 10 | 0 | `0.935` | `1.060` |
| fresh standard response cache after fix | `MTW_shadow_bin_250_medium_truth_reco_tau` | 10 | 0 | `0.935` | `1.061` |

Two separate problems are therefore present.

First, the TES shifted trees can provide shifted reconstructed observables, but
they cannot fill the `MTW_TruthMTW` migration matrix in the current builder. The
shifted input trees genuinely do not carry the truth branches needed to compute
`TruthMTW`. The TES helper recovers the fiducial event set using nominal event
masks, but it does not yet attach nominal truth-axis values to the shifted
events. This is why TES response reco projections are finite but the
`MTW_TruthMTW` matrices have zero integral and are skipped. The diagnostic also
showed that TES entries match nominal event keys, so this is fixable by adding a
truth-value lookup keyed by `(mcChannel,eventNumber)`.

Second, the response-systematic plot near `100%` was caused by invalid
efficiency-weight response histograms in the existing main response cache. The
root cause was not the allowed trigger/reconstruction/ELEOLR tau-efficiency
weight formula itself. The fresh diagnostic showed that these allowed
efficiency variations are physically sized (`0.935-1.061` of nominal) when the
response cache is rebuilt cleanly.

The actual code bug was in histogram booking: `Dataset.init_sys()` correctly
builds `self.eff_sys_set` after applying `skip_sys`, but
`Dataset.gen_all_histograms()` previously looped over every
`weight_TAUS_TRUEHADTAU_EFF_*` branch again when filling 1D and 2D systematic
histograms. That reintroduced skipped raw `JETID`/`RNNID` branches and wrote
order-one per-event factors as if they were full luminosity-normalised event
weights. The current main response cache still contains those old pathological
histograms, which is why it continues to show `~60x` variations until rebuilt.

Interpretation:
the efficiency-weight response issue is now a solved code-booking problem, not
a physics uncertainty. The fix in `src/dataset.py` makes both the 1D and 2D
systematic histogram loops use the filtered `self.eff_sys_set`, so skipped
`JETID`/`RNNID` branches are no longer written. A fresh response build confirms
that the remaining allowed efficiency variations are well normalised.

The existing `outputs/analysis_shadow_unfold/response/root/wtaunu_had.root`
cache is stale for this purpose and must be regenerated before using response
systematic plots. The TES response-matrix issue remains separate: the shifted
TES reco histograms exist, but the full `MTW_TruthMTW` matrices are still empty.

Recommendation:
before the next full-systematics run:

1. Rebuild the main response cache from scratch with the fixed
   `src/dataset.py`.
2. Keep the response-normalisation guard in `analysis_shadow_unfold.py`; it is
   still useful as a safety check against stale or malformed response inputs.
3. For TES, attach or recover the nominal truth-axis value for each shifted
   event if a full varied migration matrix is required. A full TES response
   matrix cannot be obtained from the current shifted tree alone, but the
   event-key diagnostic indicates that a nominal truth lookup should be
   feasible.
4. After the response cache is regenerated, rerun
   `run/2017/validations/validate_response_weight_normalisation.py` in
   cache-only mode to confirm that the main response cache now matches the
   fresh diagnostic.

Follow-up implementation:
The response-systematic loop now rejects varied response objects whose reco or
matrix normalization is grossly incompatible with the nominal response. This
protects against stale or malformed response inputs. The underlying
efficiency-weight booking bug has also been fixed in `src/dataset.py`: the
systematic histogram loops now iterate over the filtered `self.eff_sys_set`
rather than every raw `weight_TAUS_TRUEHADTAU_EFF_*` branch.

TES response-matrix follow-up:
The TES response-matrix problem has now been addressed in code. The shifted TES
trees contain the shifted reconstructed variables, but not the truth-axis inputs
needed to calculate `TruthMTW`. The fix is to build a nominal-event lookup keyed
by `(mcChannel, eventNumber)` and attach nominal `TruthMTW` to each shifted TES
event before filling the varied migration matrix.

Implementation:
- framework support: `src/datasetbuilder.py`
- TES response builder: `run/2017/shadow_unfold/systematics.py`
- smoke-test output:
  `outputs/validate_shadow_fakes/tes_response_lookup_build_test/response/root/wtaunu_had.root`

Smoke-test result:
| Systematic | Reco `MTW` integral | `MTW_TruthMTW` integral |
|---|---:|---:|
| `T_s1thv_NOMINAL` | 936.10 | 935.92 |
| `TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_LowPt__1down` | 861.27 | 854.62 |
| `TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_LowPt__1up` | 990.26 | 982.60 |
| `TAUS_TRUEHADTAU_SME_TES_PHYSICSLIST__1down` | 909.48 | 902.35 |
| `TAUS_TRUEHADTAU_SME_TES_PHYSICSLIST__1up` | 960.98 | 953.51 |

Interpretation:
The shifted TES matrices are no longer empty. Their matrix integrals now track
the corresponding shifted reco integrals closely, with the expected small
difference from the fiducial truth axis and migration bookkeeping. This means
TES response systematics can be rebuilt into the main response cache rather than
being skipped as structurally unavailable.

Recommendation:
Regenerate the main response cache before the next full-systematics production
run. The current fix has only been smoke-tested in a dedicated output directory;
the old `outputs/analysis_shadow_unfold/response/root/wtaunu_had.root` cache
should still be treated as stale for TES response systematics.

## Comparison With Thesis Snapshot

The thesis unfolding images live under:

`../../Documents/Thesis/images/unfolding_2017`

The current main unfolding output still differs structurally from the thesis snapshot:

| Output | Plot count | Shared with thesis | Current-only | Thesis-only |
|---|---:|---:|---:|---:|
| `outputs/unfolding_2017` | 1098 | 874 | 224 | 720 |

The current-only plots are mostly response and bin-by-bin correction diagnostics. The thesis-only plots are mostly shadow-bin systematic uncertainty plots that were not regenerated in the full-shadow test because the corresponding systematic response inputs were unavailable.

For `analysis_shadow_unfold`, there is no direct thesis snapshot equivalent. It is a diagnostic closure workflow designed to answer whether the corrected shadow-bin implementation can close.

## Current Conclusion

The corrected variable-specific shadow-bin unfolding demonstrated signal-MC closure for both `MTW` and `TauPt` during the diagnostic phase. The current cleaned production script runs the `MTW` workflow only, because `MTW` is the measurement target.

The previous failure to close was not evidence that shadow bins are intrinsically bad. It was evidence that the measured reconstructed input and the fiducial truth target were inconsistent. Once reconstructed nonfiducial signal is removed before unfolding, the closure problem essentially disappears.

The latest fake diagnostics add an important second conclusion: the remaining unfolded-data deficit is not primarily a response-closure problem. It is driven by the background-subtracted measured input, especially the size and bookkeeping of the fake estimate. The new MC-contamination-background convention fixes the clearest bookkeeping problem: data-driven fakes are no longer added on top of fake-like reconstructed MC. This improves the fiducial reco / data-signal ratio from `1.355` to `1.157` for no-shadow and from `1.384` to `1.169` for `MTW_shadow_bin_250`.

The validated 1-prong tau-width fake-source shift improves those ratios further, to `1.096` and `1.106`, but it should still be treated as a systematic-style fake-source composition variation rather than a central correction until the fake-model prescription is finalised.

The dedicated fake-validation script confirms this independently of unfolding. In the pass-ID signal selection, the prong-split fake prediction is larger than `data - MC-contamination` by factors of about `1.7` for no-shadow, `2.5` for the 200 GeV MTW shadow bin, and `2.1` for the 300 GeV MTW shadow bin.

The latest independent sideband transfer tests give the first clear direction for improving this. The nominal `TauPt` fake factor transfers well across MTW sidebands, with ratios near unity, but it transfers poorly across MET sidebands for 3-prong candidates and still meets negative 3-prong validation targets in the SR MET proxy. A follow-up MET-binned transfer test confirms that MET dependence matters, but also shows that a naive MET-binned `TauPt` fake factor is unstable in the `120 <= MET < 170` control slice, especially for 3-prong candidates.

The next physics decision is therefore split into two tracks:

1. carry the nonfiducial-signal and MC-contamination-background bookkeeping corrections into the main `unfolding_2017` workflow;
2. enable full systematic variations in the same corrected phase space;
3. test a stability-first MET-dependent fake model in the validation scripts, starting with coarser `TauPt` bins or a prong-dependent treatment in the upper-MET control slice;
4. update the thesis unfolding chapter to describe the nonfiducial signal correction, split-sample closure validation, and fake-estimate diagnostics, making clear which studies are validation-only scripts rather than part of the central production runner.

The current result is strong enough to support moving forward with the corrected unfolding implementation, but not yet strong enough to claim that the unfolded data/MC normalisation tension is understood.
