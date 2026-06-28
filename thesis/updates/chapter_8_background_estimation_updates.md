# Chapter 8 Update Notes: Background Estimation

Source audited:
- Thesis source: `/mnt/B/Uni_Stuff_Queen_Mary/Documents/Thesis/thesis.tex`
- Chapter: `Background Estimation`
- Current chapter line range: `4184-4599`
- Relevant analysis scripts:
  - `run/2017/analysis_shadow_unfold.py`
  - `src/fakes.py`
  - `run/2017/validations/validate_low_met_fake_region.py`
  - `run/2017/validations/validate_atlas_like_fake_transfer.py`
  - `run/2017/validations/validate_tau_width_composition.py`
- Relevant generated outputs:
  - `outputs/analysis_shadow_unfold/closure_summary.md`
  - `outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md`
  - `outputs/validate_shadow_fakes/atlas_like_fake_transfer/atlas_like_fake_transfer_summary.md`
  - `outputs/validate_shadow_fakes/tau_width_composition/tau_width_composition_summary.md`

Purpose:
This note records the Chapter 8 changes needed to make the background-estimation text match the current analysis. It includes both line numbers and short original-text anchors, because line numbers will shift as edits are made.

## Executive Summary

Chapter 8 needs a substantial rewrite. The current chapter still describes the older Loose-ID fake-factor workflow and several stale figure sets. The current analysis uses:

1. a Medium tau-identification signal region;
2. a low-\(\etmiss<\SI{100}{\GeV}\) fake-factor determination region;
3. prong-split fake factors derived as a function of \(p_\mathrm{T}^{\tau}\);
4. subtraction of simulated background contributions from the ID and anti-ID regions;
5. a nominal fake estimate that replaces the jet-fake-like MC component, rather than being added on top of all simulated backgrounds;
6. fake-source systematic variations from fake-factor statistics, \(\etmiss\)-window transfer, and tau-width composition.

The main physics narrative should change from "Loose-ID fake contamination motivates the fake-factor method" to "jets misidentified as hadronic tau candidates are estimated with a data-driven fake-factor method, with simulated background contributions subtracted from the determination and application regions."

## Correction 1: Opening Background Description And Vocabulary

Lines:
- `4186-4188`

Original text anchor:
> "The dominant background for this analysis are the non-\(\tau\) and neutral-current DY processes..."

> "The main sources of misidentified events come from both a multijet background ... or lepton fakes..."

Current status:
- Partly correct, but the vocabulary should be tightened.
- The current analysis distinguishes jet-to-tau misidentification, simulated electroweak/top backgrounds, and the hadronic-tau signal.
- The phrase "non-\(\tau\)" is too vague, and "lepton fakes" should not be presented as the dominant fake component without qualification.

Reason:
- ATLAS fake-factor literature treats jets misidentified as \(\tau_{\mathrm{had-vis}}\) as a distinct data-driven background because it is not reliably modelled by MC.
- The current nominal fake-factor subtraction removes simulated genuine hadronic tau, electron, and muon contamination before deriving or applying the jet-fake estimate.

Literature and analysis evidence:
- ATLAS Universal Fake Factor paper: jets misidentified as \(\tau_{\mathrm{had-vis}}\) are a sizeable background, are not reliably modelled by MC simulation, and can be estimated with fake factors measured in data control regions. Source: https://arxiv.org/abs/2502.04156.
- ATLAS high-mass \(\tau+\etmiss\) search divides background into a jet background estimated from data and other backgrounds estimated with simulation. Source: https://arxiv.org/abs/2402.16576.
- Current analysis evidence: `outputs/analysis_shadow_unfold/closure_summary.md` records `FAKE_MODEL: prong-split`, `FAKE_CONTROL_REGION: lowMET_CR`, and a pre-unfolding budget separating simulated backgrounds, jet-fake-like MC, and data-driven fakes.

Suggested correction:
- Replace the opening paragraph with a cleaner distinction between simulated backgrounds and jet-to-tau fake backgrounds.
- Do not describe the fake background as primarily "lepton fakes"; keep electrons and muons as simulated contamination subtracted from the fake-factor regions.

Suggested text:

```tex
The selected \(W\rightarrow\tau\nu\rightarrow\mathrm{hadrons}\) final state receives contributions from simulated electroweak and top-quark processes, as well as from events in which a quark- or gluon-initiated jet is reconstructed as a hadronic tau candidate. The latter contribution is referred to as the jet-to-tau fake background. It is treated separately because the rate for jets to satisfy the hadronic-tau reconstruction and identification requirements is difficult to model reliably in simulation. A data-driven fake-factor method is therefore used to estimate this contribution, with simulated background contributions subtracted in the fake-factor regions.
```

Figures/tables affected:
- None directly.

Generated outputs or scripts:
- `outputs/analysis_shadow_unfold/closure_summary.md`

## Correction 2: Loose-ID Motivation Figures Are Stale

Lines:
- `4188-4222`

Original text anchor:
> "Figures~\ref{fig:loose_bdt_scores} show the JetRNN and eBDT scores at \emph{Loose}..."

> "motivating a cut below a score of 0.4"

Current status:
- This section is stale and should not be retained as the main motivation for the current fake estimate.
- The current nominal selected tau uses Medium identification, not Loose.
- The current fake-factor method is not motivated by a new cut below a tau-ID score of `0.4`; it uses the analysis Medium ID and an anti-ID region.
- The figure has duplicate labels:
  - `fig:loose_TauRNNJetScore_stack_no_fakes_liny` appears for both 1-prong and 3-prong subfigures.
  - `fig:loose_TauBDTEleScore_stack_no_fakes_liny` appears for both 1-prong and 3-prong subfigures.

Reason:
- Chapter 7 now contains updated CR/SR stack plots for the same tau-candidate observables using Medium ID and the low-\(\etmiss\) control-region definition.
- Chapter 8 should not repeat the stale Loose-ID plots. It should introduce the fake-factor method and then show current fake-factor outputs.

Literature and analysis evidence:
- ATLAS Universal Fake Factor paper defines the fake-factor method in terms of ID and anti-ID subregions and dedicated determination regions, not as a loose-score cut study. Source: https://arxiv.org/abs/2502.04156.
- Current Chapter 7 replacement plots were generated by `run/2017/validations/validate_low_met_fake_region.py` and copied to:
  `../../Documents/Thesis/images/object_event_selections/medium_lowmet_cr_compare/`.

Suggested correction:
- Remove or heavily shorten the Loose-ID motivation paragraph and figure.
- Replace it with a short prose bridge referring back to the Chapter 7 control-region plots.

Suggested text:

```tex
The low-\(\etmiss\) control region introduced in Chapter~\ref{cha:selection} provides a high-statistics sample enriched in jets misidentified as hadronic tau candidates. The comparison of tau-identification observables in that region and in the signal region motivates the use of a data-driven transfer factor from the anti-ID region to the ID region. The fake-factor method used in this analysis is described below.
```

Figures/tables affected:
- `fig:loose_bdt_scores`, lines `4194-4222`: remove or replace.

Generated outputs or scripts:
- Replacement plots already copied to thesis images:
  - Exact Medium, prong-split, linear-y replacements for the old `fig:loose_bdt_scores` structure:
    - `images/truth_and_fakes/medium_current/medium_1prong_signal_like_sr_TauRNNJetScore_stack_no_fakes_liny.png`
    - `images/truth_and_fakes/medium_current/medium_3prong_signal_like_sr_TauRNNJetScore_stack_no_fakes_liny.png`
    - `images/truth_and_fakes/medium_current/medium_1prong_signal_like_sr_TauBDTEleScore_stack_no_fakes_liny.png`
    - `images/truth_and_fakes/medium_current/medium_3prong_signal_like_sr_TauBDTEleScore_stack_no_fakes_liny.png`
  - Inclusive Medium CR/SR linear-y comparison plots also exist under:
    - `images/object_event_selections/medium_lowmet_cr_compare/`

## Correction 3: Fake-Composition Figures Are Old And Label-Broken

Lines:
- `4192`
- `4227-4435`

Original text anchor:
> "Figures~\cref{fig:CR_fakes_contamination,fig:SR_fakes_contamination,...} show the contamination from simulated, mis-identified taus..."

> "the larges contributor to tau fakes in this analysis is electrons"

Current status:
- The old `truth_and_fakes/loose_*` plots are not aligned with the current nominal fake model.
- The text says the largest contributor is electrons, but the current nominal fake model targets jets misidentified as tau candidates and subtracts electron/muon/hadronic-tau contamination from simulation.
- Several labels are duplicated or malformed:
  - `fig:all_mc_MET_met_loose_CR_failID_fake_fractions` is reused for pass and fail CR panels.
  - `fig:all_mc_TauEta_medium_SR_failID_fake_fractions` is reused.
  - `fig:all_mc_MET_met_medium_SR_failID_fake_fractions` is reused.
  - `fig:all_mc_TauRNNJetScore_loose_SR_pNassID_fake_fractions` contains a typo.
  - `fig:medium_TauRNNJetScore_fakes_stack_TauPt_liny` later labels a tau-\(\phi\) plot.

Reason:
- These figures come from the older Loose/Medium/Tight comparison workflow. They are useful historically, but they do not document the current low-\(\etmiss\), Medium-ID, prong-split fake-factor prescription.
- The current validation outputs show fake-enrichment and transfer behaviour directly for the low-\(\etmiss\) determination region.
- The original figure style was a truth-origin fraction diagnostic, not a data-driven fake-factor closure plot. The regenerated Medium replacements therefore preserve the old plot meaning while updating the working point and region definitions.
- The control-region `\(\etmiss\)` axis should run from `10` to `100 GeV`, matching the low-\(\etmiss<\SI{100}{\GeV}\) fake-factor determination region. A previous regeneration used a `0/1` to `150 GeV` range inherited from an obsolete control-region definition; do not use those plots.

Literature and analysis evidence:
- ATLAS high-mass \(\tau+\etmiss\) search estimates the jet background with data and subtracts the remaining simulated background contributions in its jet-background control regions. Source: https://arxiv.org/abs/2402.16576.
- Current validation summary: `outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md`.
- Low-\(\etmiss\) region fake-like fractions in the current validation:
  - pass-ID 1-prong: `0.968`
  - anti-ID 1-prong: `0.998`
  - pass-ID 3-prong: `0.893`
  - anti-ID 3-prong: `0.997`

Suggested correction:
- Replace the long Loose-ID fake-composition figure sequence with the current Medium-working-point diagnostic plots, or move the old sequence to historical/supplementary material only if needed.
- The regenerated plots should keep the same meaning as the old figures: MC truth-origin fractions for reconstructed tau candidates. The Medium version separates jet, muon, electron, leptonic-tau, and hadronic-tau origins.
- The Medium control-region plots should represent the current fake-factor determination region: Medium pass-ID or fail-ID tau selection, \(p_\mathrm{T}^{\tau}>\SI{170}{\GeV}\), tau fiducial \(\eta\) acceptance, and \(\etmiss<\SI{100}{\GeV}\), without the signal-region \(\mtw\) threshold.
- Use the Medium paths listed below and update both the per-panel captions and the main figure captions. Do not keep the old Loose-working-point wording.

Suggested text:

```tex
The low-\(\etmiss\) determination region is strongly enriched in jet-to-tau fake candidates after subtracting simulated background contributions. In the anti-ID denominator, the inferred jet-to-tau component accounts for more than \(99\%\) of the data yield for both one-prong and three-prong tau candidates. This confirms that the region is suitable for measuring the fake factors used in the signal-region estimate.
```

Figures/tables affected:
- `fig:CR_fakes_contamination`, lines `4227-4268`: replace the six Loose-ID image paths with the Medium low-\(\etmiss\) determination-region images:
  - line `4231`: `truth_and_fakes/medium_CR_failID/fakes_distributions/all_mc_MET_met_medium_CR_failID_fake_fractions.png`
  - line `4237`: `truth_and_fakes/medium_CR_passID/fakes_distributions/all_mc_MET_met_medium_CR_passID_fake_fractions.png`
  - line `4244`: `truth_and_fakes/medium_CR_failID/fakes_distributions/all_mc_TauRNNJetScore_medium_CR_failID_fake_fractions.png`
  - line `4250`: `truth_and_fakes/medium_CR_passID/fakes_distributions/all_mc_TauRNNJetScore_medium_CR_passID_fake_fractions.png`
  - line `4257`: `truth_and_fakes/medium_CR_failID/fakes_distributions/all_mc_TauBDTEleScore_medium_CR_failID_fake_fractions.png`
  - line `4263`: `truth_and_fakes/medium_CR_passID/fakes_distributions/all_mc_TauBDTEleScore_medium_CR_passID_fake_fractions.png`
- `fig:SR_fakes_contamination`, lines `4270-4311`: replace the six Loose-ID image paths with the Medium signal-like application-region images:
  - line `4274`: `truth_and_fakes/medium_SR_failID/fakes_distributions/all_mc_MET_met_medium_SR_failID_fake_fractions.png`
  - line `4280`: `truth_and_fakes/medium_SR_passID/fakes_distributions/all_mc_MET_met_medium_SR_passID_fake_fractions.png`
  - line `4287`: `truth_and_fakes/medium_SR_failID/fakes_distributions/all_mc_TauRNNJetScore_medium_SR_failID_fake_fractions.png`
  - line `4293`: `truth_and_fakes/medium_SR_passID/fakes_distributions/all_mc_TauRNNJetScore_medium_SR_passID_fake_fractions.png`
  - line `4300`: `truth_and_fakes/medium_SR_failID/fakes_distributions/all_mc_TauBDTEleScore_medium_SR_failID_fake_fractions.png`
  - line `4306`: `truth_and_fakes/medium_SR_passID/fakes_distributions/all_mc_TauBDTEleScore_medium_SR_passID_fake_fractions.png`
- Suggested main caption for `fig:CR_fakes_contamination`:
  ```tex
  Fractional truth-origin composition of reconstructed tau candidates in the Medium-working-point control region, shown for \(\etmiss\), the tau jet-identification score, and the electron-rejection score. The anti-ID region is shown on the left and the ID region on the right. Each coloured component represents the MC truth origin of the reconstructed tau candidate.
  ```
- Suggested main caption for `fig:SR_fakes_contamination`:
  ```tex
  Fractional truth-origin composition of reconstructed tau candidates in the Medium-working-point signal region, shown for \(\etmiss\), the tau jet-identification score, and the electron-rejection score. The anti-ID region is shown on the left and the ID region on the right. Each coloured component represents the MC truth origin of the reconstructed tau candidate.
  ```
- `fig:medium_tight_fakes_contamination_2`, lines `4313-4374`: remove or replace.
- `fig:medium_tight_fakes_contamination_1`, lines `4376-4435`: remove or replace.

Generated outputs or scripts:
- Current fake-enrichment diagnostic plots, if used:
  - `outputs/validate_shadow_fakes/low_met_fake_region/plots/fake_enrichment/MTW_shadow_bin_300_low_met_derive_failID_1prong_TauPt_fake_enrichment.png`
  - `outputs/validate_shadow_fakes/low_met_fake_region/plots/fake_enrichment/MTW_shadow_bin_300_low_met_derive_failID_3prong_TauPt_fake_enrichment.png`
- Current Medium truth-origin replacements for `fig:CR_fakes_contamination` and `fig:SR_fakes_contamination`:
  - `outputs/truth_and_fakes_2017/plots/medium_CR_failID/fakes_distributions/`
  - `outputs/truth_and_fakes_2017/plots/medium_CR_passID/fakes_distributions/`
  - `outputs/truth_and_fakes_2017/plots/medium_SR_failID/fakes_distributions/`
  - `outputs/truth_and_fakes_2017/plots/medium_SR_passID/fakes_distributions/`
- The plot-generation script was corrected so the Medium CR uses \(\etmiss<\SI{100}{\GeV}\), no CR \(\mtw\) threshold, and CR `MET_met` binning from `10` to `100 GeV`.
- Thesis copies written to:
  - `../../Documents/Thesis/images/truth_and_fakes/medium_CR_failID/fakes_distributions/`
  - `../../Documents/Thesis/images/truth_and_fakes/medium_CR_passID/fakes_distributions/`
  - `../../Documents/Thesis/images/truth_and_fakes/medium_SR_failID/fakes_distributions/`
  - `../../Documents/Thesis/images/truth_and_fakes/medium_SR_passID/fakes_distributions/`
- Thesis copies were refreshed from the regenerated outputs on `2026-06-26`; the copied files byte-match the corresponding files under `outputs/truth_and_fakes_2017/plots/`.
- Script:
  ```bash
  pixi run python run/2017/truth_and_fakes_2017.py
  ```

## Correction 4: Fake-Factor Method Equations And Region Names

Lines:
- `4437-4456`

Original text anchor:
> "The fake factor method makes the naive assumption that the probability of a jet faking a tau in ATLAS is constant..."

> "The so-called \emph{anti-ID} region ... \(0.01 < \mathrm{JetRNN_{score}} < 0.15\)..."

> "where the `truth' subscript denotes..."

Current status:
- The basic transfer-factor idea is still correct.
- The detailed wording and equations need updating:
  - The current method uses Medium ID and anti-ID regions, not the old Loose thresholds.
  - The fake factor is derived in a low-\(\etmiss\) determination region, not simply in a CR identical to the SR except for tau ID.
  - The subtracted simulated component should be called the simulated background contribution, not "truth".
  - "take tau events" is a typo and should be "fake tau events" or, better, "jet-to-tau fake events".

Reason:
- The current nominal method follows:
  ```text
  FF = (data_ID^DR - simulated contamination_ID^DR) / (data_antiID^DR - simulated contamination_antiID^DR)
  N_fake^SR = FF * (data_antiID^SR - simulated contamination_antiID^SR)
  ```
- The simulated-contamination subtraction includes simulated genuine hadronic tau, electron, and muon matched candidates in the nominal setup.

Literature and analysis evidence:
- ATLAS Universal Fake Factor paper uses ID/anti-ID terminology and refers to fake factors measured in determination regions. Source: https://arxiv.org/abs/2502.04156.
- ATLAS high-mass \(\tau+\etmiss\) search computes the jet background from data after subtracting simulated background contributions in control regions. Source: https://arxiv.org/abs/2402.16576.
- Current implementation evidence: `src/fakes.py:31-177` and `run/2017/analysis_shadow_unfold.py:139-194`.

Suggested correction:
- Replace the old equations and prose with ID/anti-ID and determination/application-region notation.
- Avoid raw score thresholds in this chapter unless they are also present in the updated selection chapter.

Suggested text:

```tex
The fake-factor method estimates the jet-to-tau fake background by measuring a transfer factor between anti-ID and ID tau candidates in a fake-enriched determination region. The method is applied separately for one-prong and three-prong tau candidates. In each \(p_\mathrm{T}^{\tau}\) bin, the fake factor is defined as
\begin{equation}
    \mathrm{FF}_{i,j} =
    \frac{
        N^{\mathrm{data}}_{\mathrm{ID},i,j,\mathrm{DR}}
        - N^{\mathrm{sim\,contam}}_{\mathrm{ID},i,j,\mathrm{DR}}
    }{
        N^{\mathrm{data}}_{\mathrm{antiID},i,j,\mathrm{DR}}
        - N^{\mathrm{sim\,contam}}_{\mathrm{antiID},i,j,\mathrm{DR}}
    },
\end{equation}
where \(i\) labels the \(p_\mathrm{T}^{\tau}\) bin and \(j\) labels the tau charged-track multiplicity. The simulated-contamination term contains the simulated background contribution in the corresponding region.

The signal-region jet-to-tau fake estimate is then obtained by applying the fake factor to anti-ID events in the signal-region phase space:
\begin{equation}
    N^{\mathrm{jet\rightarrow\tau}}_{\mathrm{ID},i,j,\mathrm{SR}} =
    \mathrm{FF}_{i,j}
    \left(
        N^{\mathrm{data}}_{\mathrm{antiID},i,j,\mathrm{SR}}
        - N^{\mathrm{sim\,contam}}_{\mathrm{antiID},i,j,\mathrm{SR}}
    \right).
\end{equation}
```

Figures/tables affected:
- None directly.

Generated outputs or scripts:
- `src/fakes.py`
- `outputs/analysis_shadow_unfold/closure_summary.md`

## Correction 5: Current Nominal Determination Region

Lines:
- `4445`
- `4465`

Original text anchor:
> "we have labelled this region the Control Region (CR), which is defined in Table~\ref{tab:event_selection}"

> "executed for this analysis for across two event observables - the \(W\) transverse mass, \(\mtw\), and the tau transverse momentum..."

Current status:
- The current nominal fake-factor determination region is low-\(\etmiss\), not the old signal-like control region.
- The current nominal fake-factor source variable is \(p_\mathrm{T}^{\tau}\), while \(\mtw\) is the unfolded measurement variable.
- The fake estimate is prong-split.

Reason:
- The low-\(\etmiss\) region gives high statistics and is strongly fake-enriched.
- Current validation shows the low-\(\etmiss\) fake factor transfers well to the \(m_T^W\ge\SI{350}{\GeV}\), \(\etmiss<\SI{170}{\GeV}\) validation target:
  - 1-prong prediction/target: `1.006`
  - 3-prong prediction/target: `0.988`
- The high-\(\etmiss\) signal-like validation is statistically fragile, especially for 3-prong candidates after simulated-contamination subtraction.

Literature and analysis evidence:
- ATLAS high-mass \(\tau+\etmiss\) search uses low-\(\etmiss\) control regions for dijet-enriched transfer-factor measurement, with background-contamination subtraction. Source: https://arxiv.org/abs/2402.16576.
- Current validation summary: `outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md`.
- Current production summary: `outputs/analysis_shadow_unfold/closure_summary.md` records `FAKE_CONTROL_REGION: lowMET_CR`, `FAKE_FACTOR_SOURCE: TauPt`, and `FAKE_MODEL: prong-split`.

Suggested correction:
- Add a clear paragraph specifying the current fake-factor determination region and source variable.

Suggested text:

```tex
The nominal fake-factor determination region used in this analysis is defined by the same tau baseline and \(p_\mathrm{T}^{\tau}>\SI{170}{\GeV}\) requirement as the signal region, but with \(\etmiss<\SI{100}{\GeV}\) and without the signal-region \(\mtw\) requirement. This low-\(\etmiss\) region is enriched in jets misidentified as hadronic tau candidates. Fake factors are derived as a function of \(p_\mathrm{T}^{\tau}\), separately for one-prong and three-prong tau candidates, and then applied to the anti-ID signal-region events to estimate the jet-to-tau fake contribution to the \(\mtw\) spectrum.
```

Figures/tables affected:
- `fig:medium_FF_histograms_MTW`, line `4478`: old figure should not be presented as nominal.
- `fig:medium_FF_histograms_TauPt`, line `4484`: replace with a current \(p_\mathrm{T}^{\tau}\)-binned fake-factor figure if keeping this section.

Generated outputs or scripts:
- Current fake-factor comparison plot:
  - `outputs/validate_shadow_fakes/plots/no_shadow_bin/fake_factors/no_shadow_bin_lowMET_TauPt_prong_fake_factors.png`
- Do not use `outputs/validate_shadow_fakes/plots/no_shadow_bin/fake_factors/no_shadow_bin_TauPt_prong_fake_factors.png` as the nominal Chapter 8 fake-factor plot. That file shows the old non-low-\(\etmiss\) diagnostic fake factor and contains the two negative 3-prong bins.

## Correction 6: Flat 10 Percent Systematic Is No Longer The Current Treatment

Lines:
- `4467`

Original text anchor:
> "A flat 10\% systematic uncertainty is applied on fake estimate..."

Current status:
- This is no longer correct.
- The current analysis implements fake-source uncertainty components:
  - fake-factor statistical uncertainty;
  - \(\etmiss\)-window transfer envelope;
  - tau-width composition envelope.

Reason:
- The current fake uncertainties are derived from the active fake-factor machinery rather than assigned as a single flat 10% normalization.
- The \(\etmiss\)-window and tau-width components are motivated by transfer/composition effects between the determination region and application region.

Literature and analysis evidence:
- ATLAS high-mass \(\tau+\etmiss\) search assigns uncertainties for modified \(\etmiss\) control-region definitions, background-contamination subtraction, quark/gluon differences, and transfer-factor extrapolation. Source: https://arxiv.org/abs/2402.16576.
- ATLAS Universal Fake Factor paper reports fake-factor uncertainties depending on tau \(p_\mathrm{T}\) and charged-particle multiplicity. Source: https://arxiv.org/abs/2502.04156.
- Current output: `outputs/analysis_shadow_unfold/closure_summary.md`, section `Fake-source systematic envelopes`.

Suggested correction:
- Replace the flat-uncertainty sentence with a short description of the current fake-source uncertainty treatment.
- Detailed numerical uncertainty results can remain in the uncertainties chapter, but Chapter 8 should no longer claim a flat 10% fake uncertainty.

Suggested text:

```tex
The jet-to-tau fake estimate is accompanied by dedicated fake-source uncertainty variations. The statistical uncertainty in the fake factors is propagated bin-by-bin. The dependence on the low-\(\etmiss\) determination-region definition is assessed by repeating the estimate in alternate \(\etmiss\) windows and taking an envelope. A separate tau-width composition variation probes differences between the fake-like tau-candidate population in the determination and application regions.
```

Figures/tables affected:
- None directly, unless adding a summary fake-uncertainty figure.

Generated outputs or scripts:
- `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/fake_source_systematics/no_shadow_bin_MTW_4iter_combined_fake_source_uncertainty.png`
- `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/fake_source_systematics/MTW_shadow_bin_250_MTW_4iter_combined_fake_source_uncertainty.png`

## Correction 7: Results Section Should Use Current Fake-Factor Outputs

Lines:
- `4490-4509`

Original text anchor:
> "The \(\mtw\)-binned estimation predicts zero jet fakes at \(\SI{600}{\GeV}\)..."

> "This motivation is used for keeping the \(\taupt\) binning as default..."

Current status:
- The conclusion that \(p_\mathrm{T}^{\tau}\)-binned fake factors are the nominal choice is still directionally correct.
- The explanation should be updated: the current nominal fake model is already \(p_\mathrm{T}^{\tau}\)-binned, prong-split, and used for the \(\mtw\) measurement.
- Old plots under `analysis_main/medium/fake_factors/` should be replaced with current validation outputs.

Reason:
- The current fake-factor workflow is tied to the corrected shadow-unfolding analysis and to the low-\(\etmiss\) determination region.
- Current outputs include fake-factor plots, transfer-validation plots, and pre-unfolding budget summaries.

Literature and analysis evidence:
- ATLAS high-mass \(\tau+\etmiss\) search measures transfer factors in \(p_\mathrm{T}^{\tau_{\mathrm{had-vis}}}\) intervals and separately for 1-prong and 3-prong candidates. Source: https://arxiv.org/abs/2402.16576.
- Current fake-factor validation: `outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md`.
- Current fake-factor plot:
  `outputs/validate_shadow_fakes/low_met_fake_region/plots/fake_factors/no_shadow_bin_medium_TauPt_lowMET_prong_fake_factors.png`.
  This is generated by `run/2017/validations/validate_low_met_fake_region.py` using the standard analysis plotting style.

Suggested correction:
- Replace the old discussion comparing \(\mtw\)- and \(\taupt\)-binned fake factors with a concise current statement.

Suggested text:

```tex
The fake factors are derived in bins of \(p_\mathrm{T}^{\tau}\), separately for one-prong and three-prong tau candidates. This choice follows the strong dependence of the jet-to-tau misidentification rate on the tau-candidate transverse momentum and charged-track multiplicity. The resulting prong-split fake factors are applied to the anti-ID signal-region events to predict the jet-to-tau contribution to the \(\mtw\) spectrum.
```

Figures/tables affected:
- `fig:medium_MTW_FF_prong_compare`, line `4497`: remove or mark as historical.
- `fig:medium_TauPt_FF_prong_compare`, line `4503`: replace with current plot.

Generated outputs or scripts:
- Recommended current figure:
  - `outputs/validate_shadow_fakes/low_met_fake_region/plots/fake_factors/no_shadow_bin_medium_TauPt_lowMET_prong_fake_factors.png`
- Thesis copy:
  - `truth_and_fakes/medium_current/medium_taupt_prong_split_fake_factors.png`
- Do not use the older `no_shadow_bin_TauPt_prong_fake_factors.png` plot for the current nominal method; it is the non-low-\(\etmiss\) diagnostic fake factor.

## Correction 8: Final Data-MC Comparison Figures Are Stale And Belong Elsewhere

Lines:
- `4511-4598`

Original text anchor:
> "Comparisons between the full MC signal + background + fakes estimate and data..."

> "Full histograms comparisons between data and MC signal, background and fakes estimate..."

Current status:
- These figures are old `analysis_main/medium` outputs and predate the corrected background bookkeeping.
- The current unfolding input subtracts simulated backgrounds and data-driven fakes, while keeping an all-MC-background subtraction only as a diagnostic.
- The old phrase "full MC signal + background + fakes estimate" risks implying that data-driven fakes are added on top of all MC fake-like backgrounds, which is exactly the bookkeeping problem that was corrected.

Reason:
- The corrected nominal pre-unfolding input is:
  ```text
  data - simulated backgrounds - data-driven fakes - nonfiducial signal
  ```
- The current closure summary explicitly separates:
  - all MC background;
  - simulated backgrounds;
  - jet-fake-like MC background;
  - data-driven fakes.

Literature and analysis evidence:
- ATLAS high-mass \(\tau+\etmiss\) search separates jet background estimated from data and other backgrounds estimated from simulation. Source: https://arxiv.org/abs/2402.16576.
- Current pre-unfolding budget: `outputs/analysis_shadow_unfold/closure_summary.md`.
- Pre-unfolding stack composition validation:
  - `outputs/validate_shadow_fakes/preunfolding_stack_composition/plots/no_shadow_bin_preunfolding_stack_composition.png`
  - `outputs/validate_shadow_fakes/preunfolding_stack_composition/plots/MTW_shadow_bin_250_preunfolding_stack_composition.png`

Suggested correction:
- Remove or replace the old full-stack comparison figures in Chapter 8.
- If the chapter needs a plot of the corrected background bookkeeping, use the pre-unfolding stack composition plots or a compact table from `closure_summary.md`.

Suggested text:

```tex
In the final background bookkeeping, the data-driven jet-to-tau fake estimate is not added on top of the full simulated background stack. Instead, the simulated backgrounds are subtracted separately and the data-driven estimate is used for the jet-fake component. This avoids double-counting the jet-fake-like part of the simulated background prediction.
```

Figures/tables affected:
- `fig:data_mc_comparisons_with_fakes`, lines `4511-4553`: remove or replace.
- `fig:stacks_with_fakes_nostat_medium`, lines `4555-4598`: remove or replace.

Generated outputs or scripts:
- Current bookkeeping plots:
  - `outputs/validate_shadow_fakes/preunfolding_stack_composition/plots/no_shadow_bin_preunfolding_stack_composition.png`
  - `outputs/validate_shadow_fakes/preunfolding_stack_composition/plots/MTW_shadow_bin_250_preunfolding_stack_composition.png`

## Correction 9: Add A Compact Current Fake-Estimate Summary Table

Lines:
- Suggested insertion after the fake-factor equations, around `4456`, or at the start of `Fake Jet Estimation` around `4461`.

Original text anchor:
> "Finally, the estimation for the number of fake tau events in the signal region..."

Current status:
- Chapter 8 currently has many figures but no compact table that states the current fake estimate and bookkeeping.

Reason:
- The current corrected analysis has a concise numerical budget that is easier to defend than several stale plot blocks.
- This also helps the reader understand why the fake estimate is subtracted as the jet-fake component rather than added to all MC backgrounds.

Literature and analysis evidence:
- `outputs/analysis_shadow_unfold/closure_summary.md`, section `Pre-unfolding budget`.

Suggested correction:
- Add a small table for the two central configurations, or include these numbers in prose.

Suggested text:

```tex
\begin{table}[h!]
    \centering
    \begin{tabular}{lrrrr}
        Region & Data & Simulated backgrounds & Jet-fake estimate & Signal input after background subtraction \\
        \hline\hline
        No shadow bin & 1351.0 & 245.5 & 226.1 & 774.8 \\
        \(\mtw\)-shadow response & 1428.0 & 264.1 & 243.8 & 799.3 \\
    \end{tabular}
    \caption{Pre-unfolding background-subtraction budget for the corrected nominal \(d\sigma/d\mtw\) analysis. The signal input is formed after subtracting simulated backgrounds, the data-driven jet-to-tau fake estimate, and the nonfiducial signal contribution.}
    \label{tab:fake_budget_current}
\end{table}
```

Figures/tables affected:
- New table, if desired.

Generated outputs or scripts:
- `outputs/analysis_shadow_unfold/closure_summary.md`

## Chapter 8 Figure And Table Checklist

This checklist is an inventory only; each correction above explains what to do.

- Copied Medium-only replacement images are now available in the thesis image tree under:
  ```text
  /mnt/B/Uni_Stuff_Queen_Mary/Documents/Thesis/images/truth_and_fakes/medium_current/
  ```

- Recommended direct replacements:

| Thesis line(s) | Existing image/caption | Action | New image path in thesis | Caption update |
| --- | --- | --- | --- | --- |
| `4198`, `4204`, `4211`, `4217`; caption `4221` | Loose-ID JetRNN/eBDT score plots under `analysis_main/loose/no_fakes/...` | Replace with Medium, signal-like, prong-split, linear-y plots. These are the closest updated replacement for the original `fig:loose_bdt_scores` layout. | `truth_and_fakes/medium_current/medium_1prong_signal_like_sr_TauRNNJetScore_stack_no_fakes_liny.png`, `truth_and_fakes/medium_current/medium_3prong_signal_like_sr_TauRNNJetScore_stack_no_fakes_liny.png`, `truth_and_fakes/medium_current/medium_1prong_signal_like_sr_TauBDTEleScore_stack_no_fakes_liny.png`, `truth_and_fakes/medium_current/medium_3prong_signal_like_sr_TauBDTEleScore_stack_no_fakes_liny.png` | Suggested caption: `JetRNN and electron-BDT score distributions for one-prong and three-prong tau candidates at the Medium tau-identification working point in the signal-like application region. The disagreement between data and the simulated background prediction in the fake-enriched parts of these distributions motivates the data-driven jet-to-tau fake estimate.` |
| `4231`, `4237`, `4244`, `4250`, `4257`, `4263`; caption `4267` | Loose CR fake-composition plots | Replace with a compact Medium low-\(\etmiss\) fake-enrichment figure, preferably anti-ID only. | `truth_and_fakes/medium_current/medium_lowmet_antiid_1prong_taupt_fake_enrichment.png` and `truth_and_fakes/medium_current/medium_lowmet_antiid_3prong_taupt_fake_enrichment.png` | Suggested caption: `Jet-to-tau fake enrichment in the low-\(\etmiss\) anti-ID determination region for one-prong and three-prong tau candidates at the Medium tau-identification working point. Simulated background contributions are subtracted before deriving the fake factors.` |
| `4274`, `4280`, `4287`, `4293`, `4300`, `4306`; caption `4310` | Loose SR fake-composition plots | Remove, or replace only if explicitly comparing ID and anti-ID fake enrichment. | Optional: `truth_and_fakes/medium_current/medium_lowmet_id_1prong_taupt_fake_enrichment.png` and `truth_and_fakes/medium_current/medium_lowmet_id_3prong_taupt_fake_enrichment.png` | If used, caption as `Jet-to-tau fake enrichment in the low-\(\etmiss\) ID determination region...`; do not call it the signal region. |
| `4317`, `4323`, `4329`, `4336`, `4342`, `4348`, `4355`, `4361`, `4367`; caption `4373` | Loose/Medium/Tight fake-composition comparison | Remove. This is no longer needed for a Medium-only chapter. | No direct replacement needed. | Remove caption with the figure block. |
| `4380`, `4386`, `4392`, `4399`, `4405`, `4411`, `4418`, `4424`, `4430`; caption `4434` | Second Loose/Medium/Tight fake-composition comparison | Remove. This is no longer needed for a Medium-only chapter. | No direct replacement needed. | Remove caption with the figure block. |
| `4479`; caption `4480` | Old \(\mtw\)-binned intermediate fake-factor plot | Remove. The nominal fake factors are not \(\mtw\)-binned. | No direct replacement needed. | Remove caption/subfigure. |
| `4485`; caption `4486` | Old \(\taupt\)-binned intermediate fake-factor plot | Replace with current prong-split fake-factor plot. | `truth_and_fakes/medium_current/medium_taupt_prong_split_fake_factors.png` | Suggested caption: `Fake factors derived as a function of \(p_\mathrm{T}^{\tau}\), separately for one-prong and three-prong tau candidates, using the Medium tau-identification working point and the low-\(\etmiss\) determination region.` |
| `4498`; caption `4499` | Old \(\mtw\)-binned fake-factor comparison | Remove. This is no longer the nominal fake-factor observable. | No direct replacement needed. | Remove caption/subfigure. |
| `4504`; caption `4505` | Old \(\taupt\)-binned fake-factor comparison | Replace with current prong-split fake-factor plot if not already used at line `4485`. | `truth_and_fakes/medium_current/medium_taupt_prong_split_fake_factors.png` | Same caption as above. |
| `4515`, `4521`, `4528`, `4534`, `4541`, `4547`; caption `4552` | Old fake-estimate comparison plots with obsolete full-stack bookkeeping | Replace with corrected pre-unfolding bookkeeping plot, or replace the whole figure with the compact budget table in Correction 9. | `truth_and_fakes/medium_current/medium_no_shadow_preunfolding_background_bookkeeping.png` | Suggested caption: `Pre-unfolding background-subtraction budget for the nominal no-shadow \(d\sigma/d\mtw\) selection. The data-driven jet-to-tau fake estimate is used for the jet-fake component, while the remaining backgrounds are estimated from simulation.` |
| `4559`, `4565`, `4572`, `4578`, `4585`, `4592`; caption `4597` | Old full-stack data/MC/fakes plots | Remove or replace with the same corrected bookkeeping plot. If discussing the shadow-response configuration, use the shadow-bin bookkeeping plot. | Optional: `truth_and_fakes/medium_current/medium_mtw_shadow250_preunfolding_background_bookkeeping.png` | Suggested caption: `Pre-unfolding background-subtraction budget for the \(\mtw\)-shadow response configuration. This configuration is used to model migration near the nominal \(\mtw\) threshold.` |

- `fig:loose_bdt_scores`, lines `4194-4222`: stale Loose-ID figure; remove or replace.
- `fig:CR_fakes_contamination`, lines `4227-4268`: replace with Medium low-\(\etmiss\) determination-region images under `truth_and_fakes/medium_CR_*`.
- `fig:SR_fakes_contamination`, lines `4270-4311`: replace with Medium signal-like application-region images under `truth_and_fakes/medium_SR_*`.
- `fig:medium_tight_fakes_contamination_2`, lines `4313-4374`: stale and label-problematic; remove or replace.
- `fig:medium_tight_fakes_contamination_1`, lines `4376-4435`: stale and label-problematic; remove or replace.
- `fig:medium_FF_histograms_MTW`, lines `4475-4488`: old intermediate; remove or replace.
- `fig:medium_FF_histograms_TauPt`, lines `4475-4488`: old intermediate; replace if keeping an intermediate figure.
- `fig:medium_MTW_FF_prong_compare`, lines `4494-4507`: old comparison; not nominal.
- `fig:medium_TauPt_FF_prong_compare`, lines `4494-4507`: replace with current \(p_\mathrm{T}^{\tau}\)-binned prong-split fake-factor output.
- `fig:data_mc_comparisons_with_fakes`, lines `4511-4553`: stale full-stack comparison; remove or replace with corrected bookkeeping plot.
- `fig:stacks_with_fakes_nostat_medium`, lines `4555-4598`: stale full-stack comparison; remove or replace with corrected bookkeeping plot.
- Proposed new table `tab:fake_budget_current`: optional compact summary from the current closure summary.

## Immediate To-Do List

1. Rewrite the opening background-estimation paragraphs using Corrections 1-2.
2. Replace the fake-factor equations and terminology using Correction 4.
3. Update the nominal method description to say: Medium ID, low-\(\etmiss\) determination region, \(p_\mathrm{T}^{\tau}\)-binned, prong-split fake factors.
4. Remove the flat 10% fake-systematic sentence and replace it with the current fake-source uncertainty paragraph.
5. Remove or replace stale Loose-ID figure blocks. Do not keep the old figures as current evidence.
6. Add a compact current budget table if space allows.
7. Recompile the thesis and check for duplicate figure labels in Chapter 8 before moving to Chapter 9.
