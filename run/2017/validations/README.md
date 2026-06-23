# 2017 Validation Scripts

These scripts are focused validation checks for the shadow-bin fake-estimate studies.
They keep all generated outputs under:

```text
outputs/validate_shadow_fakes/
```

The historical outputs in that directory are intentionally preserved so they can be
referenced later.

## Scripts

| Script | Purpose | Expected runtime |
|---|---|---:|
| `validate_fake_transfer.py` | Cache-only smoke test for the current fake-transfer issue. It checks `MTW_shadow_bin_300`, `met_cr_split`, 1- and 3-prong, using existing sideband ROOT outputs. | seconds |
| `validate_met_binned_transfer.py` | Focused follow-up test: derive/apply `TauPt` fake factors in two MET slices while validating across an independent MTW sideband. First run may build a small cache. | minutes first run, seconds cached |
| `validate_low_met_fake_region.py` | Tests an ATLAS-like low-MET fake-enriched fake-factor derivation against the current MTW-shadow CR method. It uses `MTW_shadow_bin_300`, `MTW` only, medium ID, 1- and 3-prong, no systematics, no unfolding. | minutes first run, seconds cached |
| `validate_atlas_like_fake_transfer.py` | Tests the ATLAS tau+MET-inspired follow-up: derive prong-split `TauPt` fake factors in low-MET windows `[0,100]`, `[30,100]`, `[50,100]`, `[70,100]`, `[0,150]`, then validate in high-MET imbalanced `TauPt/MET < 0.7` regions. | minutes first run, seconds cached |
| `validate_tau_width_composition.py` | Tests whether tau track-width branches (`TauTrackWidthPt1000PV`, `TauTrackWidthPt500PV`, `TauTrackWidthPt1000TV`, `TauTrackWidthPt500TV`) indicate fake-source composition differences between the low-MET fake-factor denominator and high-MET anti-ID application regions. | minutes first run, seconds cached |
| `validate_tau_width_reweighting.py` | Diagnostic follow-up to the tau-width composition check. It applies `TauTrackWidthPt1000PV` shape-ratio weights to the high-MET anti-ID fake estimate to test whether a width-composition systematic moves the prediction materially. | minutes first run, seconds cached |
| `validate_3prong_nonfake_subtraction.py` | Cache-only breakdown of the pass-ID `data - nonfake MC` target by prong and MC component. It diagnoses why the high-MET 3-prong target is negative. | seconds |
| `validate_prong_balance_scale.py` | Cache-only implied `wtaunu_had` prong scale-factor diagnostic after subtracting fakes and other nonfake MC. Tests whether a stable 3-prong correction is defensible. | seconds |
| `validate_prong_balance_thresholds.py` | Focused high-MET threshold extension of the implied `wtaunu_had` prong scale-factor check. It keeps the `300-350 GeV` MTW shadow interval in the histogram binning, then tests whether the 3-prong imbalance appears in the shadow interval or only in the broader high-MTW region. | minutes first run, seconds cached |
| `validate_fake_health.py` | Cheap fake-factor source-bin health summary from cached validation histograms. | seconds |

## Next Script To Run

The next validation to run for the fake-source reweighting diagnostic is:

```bash
pixi run python run/2017/validations/validate_tau_width_reweighting.py
```

It writes its summary to:

```text
outputs/validate_shadow_fakes/tau_width_reweighting/tau_width_reweighting_summary.md
```
