# Chapter 7 Update Notes: Object And Event Selections

Source audited:
- Thesis source: `/mnt/B/Uni_Stuff_Queen_Mary/Documents/Thesis/thesis.tex`
- Chapter: `Object and Event Selections`
- Current chapter line range: `3815-4157`
- Current analysis evidence:
  - `run/2017/analysis_shadow_unfold.py`
  - `run/2017/shadow_unfold/selections.py`
  - `run/2017/shadow_unfold/models.py`
- Current validation outputs:
  - `outputs/analysis_shadow_unfold/closure_summary.md`
  - `outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md`
  - `outputs/validate_shadow_fakes/atlas_like_fake_transfer/atlas_like_fake_transfer_summary.md`

Purpose:
This note records the Chapter 7 changes needed to make the object and event selection text match the current 2017 shadow-unfolding analysis. It is a thesis-writing note, not a production-analysis change.

## Executive Summary

Chapter 7 needs a more substantial update than Chapter 5. The object-selection material can mostly remain as reconstruction and object-definition context, but the event-selection and tau-ID sections must be corrected to match the current analysis:

1. Clarify that the final selection is a hadronic-tau plus missing-transverse-momentum selection built from the reconstructed objects described in the chapter.
2. Keep the 2017 trigger discussion, but avoid implying that all Run-2 trigger rows enter this measurement.
3. Correct the tau section: the nominal selected tau identification is now Medium, while the anti-ID region is the Medium-fail region with a very-loose RNN floor used for the fake-factor method.
4. Correct the signal-region thresholds: the nominal no-shadow reconstructed and fiducial selection is \(p_\mathrm{T}^{\tau}>\SI{170}{\GeV}\), \(\mtw>\SI{350}{\GeV}\), and \(\etmiss>\SI{170}{\GeV}\), not \(\mtw>\SI{150}{\GeV}\) and \(\etmiss>\SI{150}{\GeV}\).
5. Correct the fake-factor control region: the active nominal fake-factor determination region is a low-\(\etmiss\) fake-enriched region with \(p_\mathrm{T}^{\tau}>\SI{170}{\GeV}\), the same tau \(\eta\) acceptance, and \(\etmiss<\SI{100}{\GeV}\). It does not impose the signal-region \(\mtw\) threshold.
6. Replace or remove the old Loose-ID control-region comparison figure block. Those figures are stale and no longer represent the current Medium-ID, low-\(\etmiss\), prong-split fake-factor workflow.

## Correction 1: Derivation And Analysis Chain Scope

Lines:
- `3817-3823`

Current status:
- The high-level DAOD/software description is still useful context.
- The chapter should describe the analysis chain without dwelling on which software layer applies each object or event selection.
- The current text is too implementation-facing for a thesis chapter on object and event selections.

Reason:
- A thesis reader needs to understand the physics objects, selected phase space, and analysis regions, not the division of labour between Athena, DTA processing, and later analysis scripts.
- The object definitions and event selections should be presented as a single coherent analysis chain.

Literature and analysis evidence:
- Analysis evidence: `run/2017/shadow_unfold/selections.py:6-40` defines the active reconstructed selection cuts used by the shadow-unfolding workflow.
- Analysis evidence: `run/2017/analysis_shadow_unfold.py:578-673` constructs the measured and response `Analysis` objects from the current selection dictionaries.
- Existing thesis citations to the Athena and DAOD machinery can remain for the derivation description.

Suggested correction:
- Add a clarifying sentence at the end of the software paragraph.
- Do not mention Python scripts, ROOT ntuples, or code variable names in the thesis prose.

Suggested text:

```tex
The reconstructed objects described below form the inputs to the event selection, fake-factor regions, and fiducial response definition used in the measurement.
```

Figures/tables affected:
- None.

Generated outputs or scripts:
- No regenerated figure is required.

## Correction 2: Trigger Selection And 2017 Scope

Lines:
- `3825-3853`
- Table `tab:triggers`

Current status:
- The trigger table gives useful Run-2 context, but the current measurement uses only the 2017 data sample.
- The text should describe the trigger requirement as part of the event selection, not as a software implementation detail.

Reason:
- The analysis is now explicitly a 2017 measurement.
- The event selection requires the relevant 2017 tau and missing-transverse-momentum trigger decision.
- The table can remain as Run-2 context, but the prose should say that only the 2017 rows enter this measurement.

Literature and analysis evidence:
- Analysis evidence: `run/2017/shadow_unfold/selections.py:6-12` includes the trigger requirement in the reconstructed preselection.
- The high-mass \(\tau+\etmiss\) ATLAS resonance search also uses trigger-based selections for \(\tau_{\mathrm{had-vis}}+\etmiss\) final states in a similar high-\(\mtw\) phase space: https://arxiv.org/abs/2402.16576.

Suggested correction:
- Add a sentence after the trigger table or before it.
- Keep Table `tab:triggers` if it is useful historical Run-2 context, but do not imply that the 2015, 2016, or 2018 rows are used in the final result.

Suggested text:

```tex
Although Table~\ref{tab:triggers} lists the Run-2 trigger menu considered during the wider analysis development, the measurement presented in this thesis uses only the 2017 dataset. Selected events are required to pass the relevant tau and missing-transverse-momentum trigger requirement for this data-taking period.
```

Figures/tables affected:
- `tab:triggers`, lines `3831-3853`: keep, but clarify that only 2017 enters the measurement.

Generated outputs or scripts:
- No regenerated figure is required.

## Correction 3: Light-Lepton And Photon Object Sections

Lines:
- Electron selection: `3861-3890`
- Photon selection: `3892-3913`
- Muon selection: `3915-3934`

Current status:
- The detailed electron, photon, and muon object-selection tables are acceptable as reconstructed-object definition context.
- The sentence at line `3859` mixes reconstructed object selection with truth-level language by saying light leptons are "prompt" and "dressed" in the object-selection section.
- The text around lines `3865-3867` implies that these objects actively form the final signal region, whereas the final selection is a hadronic-tau plus missing-transverse-momentum selection. Light objects matter through reconstruction, overlap removal, missing-transverse-momentum construction, and simulated-background classification.

Reason:
- The current event selection does not define an explicit electron, photon, or muon veto.
- The current fake-factor bookkeeping subtracts simulated real-object contamination in the fake-factor regions, including hadronic tau, electron, and muon matched candidates.
- Chapter 7 should not overstate the role of light objects in the final event selection.

Literature and analysis evidence:
- Analysis evidence: `run/2017/shadow_unfold/selections.py:6-40` shows the final reconstructed selection used by the current shadow-unfolding workflow.
- Analysis evidence: `run/2017/analysis_shadow_unfold.py:152-164` defines the simulated real-object contamination removed in the fake-factor method.
- Existing thesis citations to e/gamma and muon recommendations can remain for the reconstructed-object definitions.

Suggested correction:
- Replace the line `3859` paragraph with a less truth-specific statement.
- Reword the sentence around lines `3865-3867` so light objects are described as inputs to reconstruction, overlap-removal, \(\etmiss\), and background classification rather than final signal-region requirements.

Suggested text:

```tex
The following subsections summarise the reconstructed object definitions used by the analysis. Electrons, photons, muons, and jets enter through overlap removal, missing-transverse-momentum reconstruction, event cleaning, and simulated-background classification. The final signal selection itself is a hadronic-tau plus missing-transverse-momentum selection.
```

Figures/tables affected:
- `tab:electron_selection`, lines `3871-3890`: no table redraw required.
- `tab:photon_selection`, lines `3896-3913`: no table redraw required.
- `tab:muon_selection`, lines `3919-3934`: no table redraw required.

Generated outputs or scripts:
- No regenerated figure is required.

## Correction 4: Jet Selection And Unused \(b\)-Tagging In The Final Selection

Lines:
- `3936-3964`

Current status:
- The jet selection and jet-cleaning description remains useful reconstructed-object context.
- The \(b\)-tagging paragraph at line `3940` is not part of the current event selection.
- The current analysis uses bad-jet cleaning and \(\etmiss\), but it does not define a \(b\)-tagged control or signal region in the shadow-unfolding workflow.

Reason:
- The reconstructed event selection includes event and jet cleaning, while the final signal phase space is driven by the selected tau and \(\etmiss\)-based variables.
- Keeping the \(b\)-tagging paragraph without qualification may suggest that \(b\)-tagged jets are used in the event selection, which is not the case for the current analysis.

Literature and analysis evidence:
- Analysis evidence: `run/2017/shadow_unfold/selections.py:6-12` includes bad-jet cleaning in the reconstructed preselection.
- ATLAS missing-transverse-momentum reconstruction uses calibrated hard objects, including jets, and applies ambiguity-removal logic to avoid double-counting object contributions: https://arxiv.org/abs/1802.08168.
- Existing thesis jet-recommendation and jet-cleaning citations can remain for the reconstructed-object definitions.

Suggested correction:
- Keep the jet-selection table as reconstructed-object context.
- Add one sentence clarifying that \(b\)-tagging is not used to define the current signal or fake-factor regions.

Suggested text:

```tex
The \(b\)-tagging configuration is retained here as part of the general reconstructed-object definition, but no \(b\)-tagged signal or control region is used in the final \(W\rightarrow\tau\nu\rightarrow\mathrm{hadrons}\) measurement.
```

Figures/tables affected:
- `tab:jet_selection`, lines `3942-3964`: no table redraw required.

Generated outputs or scripts:
- No regenerated figure is required.

## Correction 5: Tau Identification Working Points And Nominal Medium-ID Selection

Lines:
- Tau working-point definition: `3966-3988`
- Tau baseline selection: `3990-4006`

Current status:
- The numerical working-point thresholds in Table `tab:tau_wp_def` match the working-point thresholds used in the analysis.
- The current unfolding no longer treats Loose, Medium, and Tight as equal candidate analysis working points. The nominal selection uses Medium tau identification.
- The Very Loose working point should be described as an anti-ID/fake-factor floor rather than the final baseline signal selection.
- Table `tab:tau_selection` contains a notation typo: the tau pseudorapidity row uses \(\eta^e\) rather than \(\eta^\tau\) or \(\eta^{\tau_{\mathrm{had-vis}}}\).

Reason:
- The current signal-region pass-ID selection uses the Medium tau-ID threshold.
- The fail-ID region used in the fake-factor method requires the electron-rejection threshold and a very-loose RNN floor, while failing the Medium tau-ID requirement.
- The final event selection requires \(p_\mathrm{T}^{\tau}>\SI{170}{\GeV}\), so the table's low object-level \(p_\mathrm{T}\) thresholds should not be confused with the analysis phase-space threshold.

Literature and analysis evidence:
- Analysis evidence: `run/2017/analysis_shadow_unfold.py:139-150` defines the Medium pass-ID and Medium fail-ID selections.
- Analysis evidence: `run/2017/shadow_unfold/selections.py:32-40` applies the \(p_\mathrm{T}^{\tau}\), \(\eta\), \(\mtw\), and \(\etmiss\) signal-region thresholds.
- ATLAS tau performance work documents the use of tau identification working points and dedicated tau energy calibration/performance measurements for hadronic tau candidates: https://arxiv.org/abs/1412.7086.
- The high-mass \(\tau+\etmiss\) ATLAS resonance search uses \(\tau_{\mathrm{had-vis}}\) candidates, tau-ID requirements, and high-\(p_\mathrm{T}\)/high-\(\etmiss\) event selections in a similar final state: https://arxiv.org/abs/2402.16576.

Suggested correction:
- Keep Table `tab:tau_wp_def`, but rewrite the surrounding text so it no longer says that all working points are compared for the current analysis.
- Update Table `tab:tau_selection` or add text below it to state that the final nominal selected tau uses the Medium working point.
- Correct the tau \(\eta\) notation in Table `tab:tau_selection`.

Suggested text:

```tex
This measurement uses the Medium tau-identification working point for the selected signal-region tau candidate. The Very Loose requirement is retained as a loose anti-ID floor in the fake-factor method, where events that fail the Medium requirement but pass the loose RNN-score floor define the anti-ID application region. Loose and Tight definitions were used in earlier comparison studies but are not the nominal selection for the final unfolding.
```

Suggested table edits:
- In `tab:tau_selection`, replace `\eta^e` with `\eta^{\tau_{\mathrm{had-vis}}}`.
- Replace `Working Point & Very Loose` with `Nominal selected-tau ID & Medium`.
- If retaining a row for the anti-ID/fake-factor floor, add `Anti-ID floor & Very Loose RNN-score floor, fail Medium`.

Figures/tables affected:
- `tab:tau_wp_def`, lines `3970-3988`: keep.
- `tab:tau_selection`, lines `3992-4006`: update wording and tau notation.

Generated outputs or scripts:
- No regenerated figure is required.

## Correction 6: Overlap Removal And Missing Transverse Momentum

Lines:
- Overlap removal: `4008-4029`
- Missing transverse energy: `4031-4050`

Current status:
- The overlap-removal and \(\etmiss\) sections remain useful as reconstructed-object context.
- The table reference at line `4033` is incorrect: it points to `sec:met_selection` even though the table label is `tab:met_selection`.
- The \(\etmiss\) selection section should distinguish object-level \(\etmiss\) reconstruction from the final event-level \(\etmiss>\SI{170}{\GeV}\) signal-region requirement.

Reason:
- The event selection cuts on the reconstructed \(\etmiss\) after the object-level \(\etmiss\) reconstruction.
- The low-\(\etmiss\) fake-factor control region uses the same reconstructed \(\etmiss\) definition.

Literature and analysis evidence:
- Analysis evidence: `run/2017/shadow_unfold/selections.py:32-40` applies the reconstructed signal-region \(\etmiss\) threshold.
- Analysis evidence: `run/2017/analysis_shadow_unfold.py:186-195` defines the low-\(\etmiss\) fake-enriched control region.
- ATLAS missing-transverse-momentum performance documentation describes \(\etmiss\) reconstruction from calibrated hard objects and a track-based soft term, including ambiguity removal between objects: https://arxiv.org/abs/1802.08168.
- A later Run-2 ATLAS \(\etmiss\) performance paper documents the full Run-2 \(\etmiss\) reconstruction and significance performance using \(140~\mathrm{fb}^{-1}\): https://arxiv.org/abs/2402.05858.

Suggested correction:
- Fix the table reference.
- Add a sentence after the \(\etmiss\) object-definition paragraph that the analysis-level \(\etmiss\) cuts are listed in the event-selection table.

Suggested text:

```tex
The \(\etmiss\) object definition described here is the reconstructed quantity used later in the event selection. The analysis-level requirements on this quantity, including the high-\(\etmiss\) signal region and the low-\(\etmiss\) fake-factor region, are listed in Table~\ref{tab:event_selection}.
```

Suggested reference fix:

```tex
A selection summary is given in Table~\ref{tab:met_selection}.
```

Figures/tables affected:
- `tab:olr`, lines `4012-4029`: keep.
- `tab:met_selection`, lines `4035-4050`: keep; fix the text reference at line `4033`.

Generated outputs or scripts:
- No regenerated figure is required.

## Correction 7: Event Selection, Signal Region, Control Region, And Fiducial Region

Lines:
- Event selection prose: `4052-4068`
- Selection table: `4114-4155`

Current status:
- This is the main section that must be updated.
- The prose at line `4062` says \(\mtw>\SI{150}{\GeV}\) and \(\etmiss>\SI{150}{\GeV}\), but the current nominal no-shadow analysis uses \(\mtw>\SI{350}{\GeV}\) and \(\etmiss>\SI{170}{\GeV}\).
- The control-region prose at line `4064` says the fake-factor control region has the same \(\mtw\) selection as the signal region. That is no longer the active nominal fake-factor region.
- The current active fake-factor control region is a low-\(\etmiss\) fake-enriched region with \(p_\mathrm{T}^{\tau}>\SI{170}{\GeV}\), tau \(\eta\) acceptance, and \(\etmiss<\SI{100}{\GeV}\), with no \(\mtw\) threshold.
- The table still says the nominal tau ID is `Loose/Medium/Tight`; the current nominal selected-tau ID is Medium.

Reason:
- The current analysis has consolidated to the \(d\sigma/d\mtw\) measurement with a nominal no-shadow phase space and an \(\mtw\)-shadow response variation.
- The no-shadow configuration is the nominal signal phase space. The shadow-bin configuration lowers the \(\mtw\) boundary to model migration across the nominal \(\mtw>\SI{350}{\GeV}\) lower edge.
- The low-\(\etmiss\) fake-factor control region was introduced after the original thesis draft and is now the active nominal fake-factor region.

Literature and analysis evidence:
- Analysis evidence: `run/2017/shadow_unfold/selections.py:32-56` defines the reconstructed and fiducial selection builders.
- Analysis evidence: `run/2017/analysis_shadow_unfold.py:186-195` defines the active low-\(\etmiss\) fake-factor control region.
- Analysis evidence: `run/2017/analysis_shadow_unfold.py:263-275` defines the no-shadow and \(\mtw\)-shadow configurations.
- Analysis output: `outputs/analysis_shadow_unfold/closure_summary.md` records the low-\(\etmiss\), \(p_\mathrm{T}^{\tau}\)-sourced, prong-split fake-factor model.
- The high-mass \(\tau+\etmiss\) ATLAS resonance search uses high-\(\mtw\) and high-\(\etmiss\) kinematic selections in the same broad final state, supporting the use of a high-\(\mtw\), high-\(\etmiss\) signal phase space: https://arxiv.org/abs/2402.16576.
- The ATLAS Universal Fake Factor method motivates deriving jet-to-tau fake estimates in regions enriched in jets misidentified as tau candidates and transferring them to the application region: https://arxiv.org/abs/2502.04156.

Suggested correction:
- Replace the prose at lines `4062-4066`.
- Replace Table `tab:event_selection` or split it into a nominal signal/fiducial table plus a separate fake-factor region table.
- Use "anti-ID" or "fail Medium ID" for the fake-factor denominator/application region rather than calling it simply a second tau-ID working point.

Suggested text:

```tex
The nominal signal region is defined by requiring exactly one reconstructed hadronic tau candidate passing the baseline event selection, one or three associated charged tracks, unit charge, and the Medium tau-identification working point. The selected tau must satisfy \(p_\mathrm{T}^{\tau}>\SI{170}{\GeV}\) and lie within the fiducial tau pseudorapidity acceptance, excluding the barrel--endcap transition region. The event must also satisfy \(\etmiss>\SI{170}{\GeV}\) and \(\mtw>\SI{350}{\GeV}\). These thresholds define the no-shadow-bin signal region used for the central measurement.

For the unfolding response, an additional \(\mtw\)-shadow configuration is used in which the reconstructed and fiducial \(\mtw\) threshold is lowered while keeping the same \(p_\mathrm{T}^{\tau}\), \(\eta\), and \(\etmiss\) requirements. This enlarged response phase space models migration into the nominal \(\mtw>\SI{350}{\GeV}\) region.

The fake-factor determination region is defined separately from the signal region. The nominal fake-factor control region uses the same tau baseline and \(p_\mathrm{T}^{\tau}>\SI{170}{\GeV}\) requirement, but requires \(\etmiss<\SI{100}{\GeV}\) and does not impose the signal-region \(\mtw\) threshold. The fake factors are derived separately for one- and three-prong tau candidates as a function of \(p_\mathrm{T}^{\tau}\), using Medium pass-ID and fail-ID regions.
```

Suggested table replacement:

```tex
\begin{table}[h!]
    \centering
    \begin{tabular}{ll}
        Feature & Selection \\
        \hline\hline
        Fiducial no-shadow region \\
        \hline
        Tau decay mode & Hadronic tau decay with 1 or 3 charged particles \\
        Truth tau pseudorapidity & \((|\eta^{\tau_\mathrm{had-vis}}|<1.37)\ ||\ (1.52<|\eta^{\tau_\mathrm{had-vis}}|<2.47)\) \\
        Visible truth tau momentum & \(p_\mathrm{T}^{\tau_\mathrm{had-vis}}>\SI{170}{\GeV}\) \\
        Truth transverse mass & \(\mtw>\SI{350}{\GeV}\) \\
        Truth neutrino transverse momentum & \(p_\mathrm{T}^{\nu}>\SI{170}{\GeV}\) \\
        \hline\hline
        Reconstructed signal region \\
        \hline
        Trigger & Pass tau and missing-transverse-momentum trigger requirement \\
        Event cleaning & Pass event and bad-jet cleaning \\
        Primary vertex & At least one primary vertex \\
        Tau multiplicity & Exactly one baseline hadronic tau candidate \\
        Tau charge & \(|q|=1\) \\
        Tau tracks & 1 or 3 associated charged tracks \\
        Tau pseudorapidity & \((|\eta^{\tau_\mathrm{had-vis}}|<1.37)\ ||\ (1.52<|\eta^{\tau_\mathrm{had-vis}}|<2.47)\) \\
        Tau transverse momentum & \(p_\mathrm{T}^{\tau}>\SI{170}{\GeV}\) \\
        Missing transverse momentum & \(\etmiss>\SI{170}{\GeV}\) \\
        Transverse mass & \(\mtw>\SI{350}{\GeV}\) \\
        Tau identification & Medium \\
        \hline\hline
        Fake-factor determination region \\
        \hline
        Tau baseline, charge, tracks, and pseudorapidity & Same as reconstructed signal region \\
        Tau transverse momentum & \(p_\mathrm{T}^{\tau}>\SI{170}{\GeV}\) \\
        Missing transverse momentum & \(\etmiss<\SI{100}{\GeV}\) \\
        Tau ID regions & Medium pass-ID and fail-ID anti-ID regions \\
    \end{tabular}
    \caption{Selection criteria for the fiducial no-shadow region, reconstructed signal region, and fake-factor determination region used in this analysis.}
    \label{tab:event_selection}
\end{table}
```

Figures/tables affected:
- `tab:event_selection`, lines `4114-4155`: replace or substantially edit.

Generated outputs or scripts:
- Current summary:
  ```text
  outputs/analysis_shadow_unfold/closure_summary.md
  ```
- Representative response plots:
  ```text
  outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/no_shadow_bin_MTW_response_matrix.png
  outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/MTW_shadow_bin_250_MTW_response_matrix.png
  ```

## Correction 8: Low-\(\etmiss\) Fake-Region Motivation Figure

Lines:
- Figure block: `4070-4112`
- Figure label: `fig:loose_CR_compare`

Current status:
- The figure block is stale and should not be used as-is.
- The included images are old Loose-ID, no-fakes stack plots under `analysis_main/loose/no_fakes/...`.
- The current fake-factor model is Medium-ID, prong-split, \(p_\mathrm{T}^{\tau}\)-sourced, and uses a low-\(\etmiss<\SI{100}{\GeV}\) fake-factor determination region.
- The preferred replacement is an updated version of the same style of figure already used in the thesis: control-region and signal-region stacked distributions for tau-identification observables, now using the current Medium tau-identification selection and low-\(\etmiss\) control-region definition.

Reason:
- The old images in the thesis image tree are dated `2024-12-01` and predate the current shadow-unfolding/fake-factor revisions.
- The current validation evidence for the low-\(\etmiss\) fake-factor region is in `outputs/validate_shadow_fakes/`, not in the old `analysis_main/loose/no_fakes` plot set.
- In the refreshed low-\(\etmiss\) control-region validation, the anti-ID denominator is \(99.8\%\) fake-like for 1-prong candidates and \(99.7\%\) fake-like for 3-prong candidates after subtracting simulated nonfake contamination. The pass-ID numerator is also fake-like dominated, at \(96.8\%\) for 1-prong and \(89.3\%\) for 3-prong candidates.
- The updated stack plots preserve the visual structure of the existing Chapter 7 figure while correcting the region definition, tau-identification working point, and sample bookkeeping.

Literature and analysis evidence:
- Analysis evidence: `outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md` compares the low-\(\etmiss\) fake-enriched derivation to the earlier MTW-shadow control-region construction.
- Analysis evidence: `outputs/validate_shadow_fakes/atlas_like_fake_transfer/atlas_like_fake_transfer_summary.md` tests ATLAS-like low-\(\etmiss\) transfer windows and high-\(\etmiss\) validation regions.
- The ATLAS high-mass \(\tau+\etmiss\) resonance search uses low-\(\etmiss\), jet-enriched fake-factor ingredients and validates transfer to higher-\(\etmiss\) regions: https://arxiv.org/abs/2402.16576.
- The ATLAS Universal Fake Factor paper supports using data-driven fake factors for jets misidentified as tau candidates: https://arxiv.org/abs/2502.04156.

Suggested correction:
- Remove the six-panel Loose-ID plot block.
- Replace it with a six-panel figure of the same form: the low-\(\etmiss\) control region and the signal region, each shown for the tau jet-identification score, the electron-rejection score, and the tau track multiplicity.
- Update the caption and subcaptions from Loose tau identification to Medium tau identification.
- Use the updated thesis-like stack plots for Chapter 7. The fake-enrichment and transfer-ratio plots remain useful diagnostics, but they are less visually continuous with the current thesis layout.

Suggested replacement caption:

```tex
Comparison of reconstructed tau-candidate observables in the low-\(\etmiss\) control region and the signal region after the Medium tau-identification requirement. The tau jet-identification score, electron-rejection score, and track multiplicity are shown in the two regions. The low-\(\etmiss\) region provides a high-statistics sample enriched in jets misidentified as hadronic tau candidates and is used to determine the jet-to-tau fake factors.
```

Figures/tables affected:
- `fig:loose_CR_compare`, lines `4070-4112`: remove or replace.

Generated outputs or scripts:
- Regeneration command:
  ```bash
  pixi run python run/2017/validations/validate_low_met_fake_region.py
  ```
- If the current histogram cache predates these thesis-like variables and selections, run once with `LOAD_SAVED_HISTS = False` in `run/2017/validations/validate_low_met_fake_region.py`; after the cache is rebuilt, restore `LOAD_SAVED_HISTS = True` for faster redraws.
- Current validation summary:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md
  ```
- Recommended Chapter 7 figure inputs:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/plots/thesis_like_region_stacks/MTW_shadow_bin_300_low_met_medium_CR_passID/medium_TauRNNJetScore_stack_no_fakes_log.png
  outputs/validate_shadow_fakes/low_met_fake_region/plots/thesis_like_region_stacks/MTW_shadow_bin_300_signal_like_medium_SR_passID/medium_TauRNNJetScore_stack_no_fakes_log.png
  outputs/validate_shadow_fakes/low_met_fake_region/plots/thesis_like_region_stacks/MTW_shadow_bin_300_low_met_medium_CR_passID/medium_TauBDTEleScore_stack_no_fakes_log.png
  outputs/validate_shadow_fakes/low_met_fake_region/plots/thesis_like_region_stacks/MTW_shadow_bin_300_signal_like_medium_SR_passID/medium_TauBDTEleScore_stack_no_fakes_log.png
  outputs/validate_shadow_fakes/low_met_fake_region/plots/thesis_like_region_stacks/MTW_shadow_bin_300_low_met_medium_CR_passID/medium_TauNCoreTracks_stack_no_fakes_log.png
  outputs/validate_shadow_fakes/low_met_fake_region/plots/thesis_like_region_stacks/MTW_shadow_bin_300_signal_like_medium_SR_passID/medium_TauNCoreTracks_stack_no_fakes_log.png
  ```
- Optional supporting fake-enrichment diagnostic plots, not recommended as the main Chapter 7 replacement:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/plots/fake_enrichment/MTW_shadow_bin_300_low_met_derive_failID_1prong_TauPt_fake_enrichment.png
  outputs/validate_shadow_fakes/low_met_fake_region/plots/fake_enrichment/MTW_shadow_bin_300_low_met_derive_failID_3prong_TauPt_fake_enrichment.png
  outputs/validate_shadow_fakes/low_met_fake_region/plots/fake_enrichment/MTW_shadow_bin_300_low_met_derive_passID_1prong_TauPt_fake_enrichment.png
  outputs/validate_shadow_fakes/low_met_fake_region/plots/fake_enrichment/MTW_shadow_bin_300_low_met_derive_passID_3prong_TauPt_fake_enrichment.png
  ```

## Chapter 7 Figure And Table Checklist

This checklist is only an inventory; the corrections above describe what to do.

- `tab:triggers`, lines `3831-3853`: keep, clarify that only the 2017 trigger decisions enter the final measurement.
- `tab:electron_selection`, lines `3871-3890`: keep as reconstructed-object context.
- `tab:photon_selection`, lines `3896-3913`: keep as reconstructed-object context.
- `tab:muon_selection`, lines `3919-3934`: keep as reconstructed-object context.
- `tab:jet_selection`, lines `3942-3964`: keep as reconstructed-object context; qualify that \(b\)-tagging is not used in the final event selection.
- `tab:tau_wp_def`, lines `3970-3988`: keep, but update surrounding text so Medium is the nominal selected-tau ID.
- `tab:tau_selection`, lines `3992-4006`: update tau notation and nominal selected-tau ID.
- `tab:olr`, lines `4012-4029`: keep; fix minor caption grammar if editing.
- `tab:met_selection`, lines `4035-4050`: keep; fix the reference typo at line `4033`.
- `fig:loose_CR_compare`, lines `4070-4112`: remove or replace with current low-\(\etmiss\) validation material.
- `tab:event_selection`, lines `4114-4155`: replace or substantially edit to match the current no-shadow, shadow-response, and fake-factor selections.

## Immediate To-Do List

1. Apply Corrections 1-8 as thesis text edits.
2. Correct the signal-region prose at line `4062` before any other Chapter 7 edits; it currently conflicts with both the selection table and the current analysis.
3. Replace Table `tab:event_selection` with the updated no-shadow/fake-factor table above, or split it into two clearer tables.
4. Remove or replace `fig:loose_CR_compare`; do not keep the old Loose-ID `analysis_main/loose/no_fakes` plot block as current evidence.
5. Fix the `Table~\ref{sec:met_selection}` typo to `Table~\ref{tab:met_selection}`.
6. Recompile the thesis and inspect Chapter 7 table placement, especially the long event-selection table.
