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
| `validate_fake_scale_unfolding.py` | Cache-only preservation of the old fake-scale unfolding diagnostic. It unfolds the saved `analysis_shadow_unfold` input with fake scales `0`, `0.5`, and `1.0`. | seconds |
| `validate_mc_fake_closure.py` | Cache-only preservation of the old MC-only fake closure check using the known non-true-tau MC component as the fake target. | seconds |
| `validate_split_sample_unfolding_closure.py` | Cache-only split-sample MC unfolding closure check using historical `split_response` and `split_pseudo_data` outputs. | seconds if caches exist |
| `validate_prong_model_unfolding_impact.py` | Cache-only propagated unfolding test for a prong-dependent `wtaunu_had` modelling variation. Requires prong-split response selections from the historical output or a dedicated producer. | seconds if caches exist |
| `validate_preunfolding_stack_composition.py` | Cache-only reconstructed-level stack audit. It checks whether adding data-driven fakes double-counts fake-like MC already present in the MC stack. | seconds |
| `validate_failid_fake_application_breakdown.py` | Cache-only source-bin audit of how small `TauPt` fake factors become a large SR fake prediction. It also reports the available sample-level fail-ID fake-like/nonfake split. | seconds |
| `validate_failid_truth_categories.py` | Focused SR fail-ID truth-category audit. It decomposes the `TauPt` application population into hadronic tau, leptonic tau, electron, muon, photon, and jet-like/unmatched categories for `no_shadow_bin` and `MTW_shadow_bin_250`. | minutes first run, seconds cached |
| `validate_photon_nonfake_subtraction.py` | Validation-only fake-estimate variant that adds photon-matched candidates, and optionally leptonic tau, to the MC nonfake subtraction. It tests whether photon treatment materially reduces the fake yield and reconstructed stack overshoot. | minutes first run, seconds cached |
| `validate_note_like_loose_fake_factor.py` | Validation-only comparison to the ATLAS high-mass `tau + MET` note. It builds Loose/VeryLoose-proxy fake factors in `MET < 100` and compares the `350-500` and `500-1000` GeV bins to Table 14. | minutes first run, seconds cached |

## Next Script To Run

The next validation sequence for the fake-normalisation issue is:

```bash
pixi run python run/2017/validations/validate_preunfolding_stack_composition.py
pixi run python run/2017/validations/validate_failid_fake_application_breakdown.py
pixi run python run/2017/validations/validate_failid_truth_categories.py
pixi run python run/2017/validations/validate_photon_nonfake_subtraction.py
pixi run python run/2017/validations/validate_note_like_loose_fake_factor.py
```

They write summaries to:

```text
outputs/validate_shadow_fakes/preunfolding_stack_composition/preunfolding_stack_composition_summary.md
outputs/validate_shadow_fakes/failid_fake_application_breakdown/failid_fake_application_breakdown_summary.md
outputs/validate_shadow_fakes/failid_truth_categories/failid_truth_categories_summary.md
outputs/validate_shadow_fakes/photon_nonfake_subtraction/photon_nonfake_subtraction_summary.md
outputs/validate_shadow_fakes/note_like_loose_fake_factor/note_like_loose_fake_factor_summary.md
```
