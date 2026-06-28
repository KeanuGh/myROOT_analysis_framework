# Chapter 10 Update Notes: Data-Simulation Comparisons

Source audited:
- Thesis source: `/mnt/B/Uni_Stuff_Queen_Mary/Documents/Thesis/thesis.tex`
- Chapter: `Data-Simulation Comparisons`
- Current chapter line range: `4543-5030`
- Relevant analysis scripts:
  - `run/2017/validations/validate_low_met_fake_region.py`
  - `run/2017/analysis_shadow_unfold.py`
  - `src/fakes.py`
- Relevant generated outputs:
  - `outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md`
  - `outputs/validate_shadow_fakes/low_met_fake_region/plots/current_data_mc_fakes_stacks/`
  - `outputs/analysis_shadow_unfold/closure_summary.md`
- Current thesis image copies:
  - `/mnt/B/Uni_Stuff_Queen_Mary/Documents/Thesis/images/truth_and_fakes/medium_current/current_data_mc_fakes_stacks/`

Purpose:
This note records what should be changed so Chapter 10 matches the current Medium tau-identification analysis and the corrected data-driven jet-to-tau fake estimate.

## Executive Summary

Chapter 10 still contains a substantial amount of stale material from the older `analysis_simple_2017` comparison workflow. The current chapter should be tightened around the current Medium working-point measurement:

1. Remove the uncertainty-equation and systematic-table material that currently appears at the start of Chapter 10. It is not a data-simulation comparison and duplicates Chapter 9 content.
2. Replace the old filled stack plots under `analysis_simple_2017/medium/fakes/` with the current line-overlay plots showing data, MC without the data-driven fake estimate, and MC with the data-driven jet-to-tau fake estimate.
3. Do not describe the corrected comparison as "full MC signal + background + fakes", because that wording obscures the current fake-background bookkeeping.
4. Replace the old prong-split and charge-split comparison figures with the newly regenerated Medium line-overlay plots. Current replacements now exist for the same six observables used in the thesis figure blocks.
5. Remove or regenerate the old event-count, cutflow, and pie-chart summaries. The existing values are stale and do not reflect the current low-$E_{\mathrm{T}}^{\mathrm{miss}}$, prong-split fake-factor workflow.
6. Keep the chapter focused on 2017 Medium-ID data-simulation comparisons. Loose and Tight supplementary comparisons are no longer aligned with the current nominal analysis unless they are regenerated.

## Correction 1: Opening Scope And Misplaced Uncertainty Material

Lines:
- Opening paragraph: `4545`
- Uncertainty equations and trigger/systematic tables: `4547-4620`

Original text anchor:
> "Signal and background estimations between MC and data are presented..."

> "Full systematic uncertainties for this analysis are calculated in the following way..."

Current status:
- The opening paragraph still says the figures show total uncertainties including systematic uncertainties.
- The current replacement comparison plots show statistical uncertainties on the plotted event yields, not the full experimental and fake-source uncertainty model.
- Lines `4547-4620` belong to the uncertainty discussion rather than the data-simulation comparison chapter.
- The systematic table exposes raw implementation names and duplicates the Chapter 9 uncertainty content.

Reason:
- Chapter 10 should introduce the data/MC comparison figures and what is being compared.
- The uncertainty model is already described in Chapter 9.
- The current comparison plots compare:
  - data;
  - MC signal plus simulated backgrounds before adding the data-driven fake estimate;
  - MC signal plus simulated backgrounds after adding the data-driven jet-to-tau fake estimate.
- The current fake estimate follows the corrected data-driven bookkeeping described in Chapters 8 and 9.

Literature and analysis evidence:
- The ATLAS high-mass $\tau+E_{\mathrm{T}}^{\mathrm{miss}}$ search separates the jet background estimated from data from other simulated backgrounds, supporting a comparison that distinguishes MC-only prediction from prediction including the data-driven fake estimate: https://arxiv.org/abs/2402.16576.
- The ATLAS Universal Fake Factor paper motivates data-driven fake factors for jets misidentified as hadronic tau candidates: https://arxiv.org/abs/2502.04156.
- Analysis evidence: `outputs/analysis_shadow_unfold/closure_summary.md` records the current fake model as `FAKE_CONTROL_REGION: lowMET_CR`, `FAKE_FACTOR_SOURCE: TauPt`, and `FAKE_MODEL: prong-split`.
- Analysis evidence: `outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md` lists the current data/MC plus fake-estimate comparison plots.

Suggested correction:
- Replace the opening paragraph with a shorter analysis-facing introduction.
- Remove lines `4547-4620` from Chapter 10. Do not keep the uncertainty equations, trigger table, or raw systematic table in this chapter.

Suggested text:

```tex
This chapter compares the selected 2017 data with the reconstructed prediction in the Medium tau-identification signal region. The MC prediction is shown both before and after adding the data-driven jet-to-tau fake estimate described in Chapter~\ref{cha:fakes}. The comparison is made for the reconstructed observables most directly connected to the selection and fake estimate: $\mtw$, $\etmiss$, $\taupt$, and the charged-track multiplicity of the selected tau candidate. The uncertainty treatment is described in Chapter~\ref{cha:uncertainties}; the figures in this chapter show the statistical uncertainties associated with the plotted event yields.
```

Figures/tables affected:
- Remove `tab:triggers_2017`, lines `4578-4592`, from Chapter 10.
- Remove `tab:systematics_full`, lines `4594-4620`, from Chapter 10.

Generated outputs or scripts:
- No plot regeneration is required for this wording change.

## Correction 2: Inclusive Medium Data/MC Comparison Figure

Lines:
- Figure block: `4622-4664`

Original text anchor:
> `analysis_simple_2017/medium/fakes/medium_MTW_stack_fakes_log.png`

> "Comparisons between the full MC signal + background + fakes estimate and data..."

Current status:
- The old figure block uses stale `analysis_simple_2017` filled stack plots.
- The caption uses obsolete "full MC signal + background + fakes" wording.
- Current inclusive Medium comparison plots exist for the same six variables used in the old thesis block:
  - $m_{\mathrm{T}}^W$;
  - $E_{\mathrm{T}}^{\mathrm{miss}}$;
  - tau transverse momentum;
  - tau pseudorapidity;
  - the absolute azimuthal separation between the tau candidate and missing transverse momentum;
  - tau azimuthal angle.
- Current TauRNN, electron-BDT, and charged-track-multiplicity comparison plots also exist, but they are supporting diagnostics rather than one-for-one replacements for the main Chapter 10 figure block.

Reason:
- The current corrected comparison should be a line-overlay comparison, not a filled stack.
- The two MC curves are useful because they show the effect of the data-driven jet-to-tau fake estimate directly.
- The six regenerated variables provide a one-for-one replacement of the old thesis comparison block while preserving the corrected Medium-ID, low-$E_{\mathrm{T}}^{\mathrm{miss}}$ fake-factor bookkeeping.

Literature and analysis evidence:
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py` builds the current inclusive, prong-split, and charge-split data/MC plus fake-estimate line-overlay plots.
- Current output summary: `outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md`, section `Current data/MC plus fake-estimate stacks`.
- Thesis copies already exist under `images/truth_and_fakes/medium_current/current_data_mc_fakes_stacks/`.

Suggested correction:
- Replace the whole figure block with a six-panel current inclusive Medium figure.
- Use the linear-y versions for the main chapter. Log-y versions exist and can be used only if the discussion explicitly focuses on tails.

Suggested text:

```tex
\begin{figure}[h]
    \centering
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/current_data_mc_fakes_stacks/medium_MTW_current_signal_background_fakes_liny.png}
        \caption{$\mtw$.}
        \label{fig:medium_current_mtw_data_mc_fakes}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/current_data_mc_fakes_stacks/medium_MET_met_current_signal_background_fakes_liny.png}
        \caption{$\etmiss$.}
        \label{fig:medium_current_etmiss_data_mc_fakes}
    \end{subfigure}
    \vfill
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/current_data_mc_fakes_stacks/medium_TauPt_current_signal_background_fakes_liny.png}
        \caption{$\taupt$.}
        \label{fig:medium_current_taupt_data_mc_fakes}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/current_data_mc_fakes_stacks/medium_TauEta_current_signal_background_fakes_liny.png}
        \caption{$\eta^\tau$.}
        \label{fig:medium_current_taueta_data_mc_fakes}
    \end{subfigure}
    \vfill
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/current_data_mc_fakes_stacks/medium_AbsDeltaPhi_tau_met_current_signal_background_fakes_liny.png}
        \caption{$|\Delta\phi(\tau,\etmiss)|$.}
        \label{fig:medium_current_dphitauetmiss_data_mc_fakes}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/current_data_mc_fakes_stacks/medium_TauPhi_current_signal_background_fakes_liny.png}
        \caption{$\phi^\tau$.}
        \label{fig:medium_current_tauphi_data_mc_fakes}
    \end{subfigure}
    \caption{Comparison between data and the reconstructed prediction in the Medium tau-identification signal region. The MC prediction is shown before and after adding the data-driven jet-to-tau fake estimate. The lower panels show the prediction divided by the data, and error bars show statistical uncertainties.}
    \label{fig:medium_current_data_mc_fakes}
\end{figure}
```

Figures/tables affected:
- Replace `fig:stacks_with_fakes_syst_medium`, lines `4622-4664`.
- Remove the old subfigure labels:
  - `fig:medium_MTW_fakes_stack_TauPt_liny`
  - `fig:medium_MET_met_fakes_stack_TauPt_liny`
  - `fig:medium_TauPt_fakes_stack_TauPt_liny`
  - `fig:medium_TauEta_fakes_stack_TauPt_liny`
  - `fig:medium_TauNCoreTracks_stack_fakes_liny`
  - `fig:medium_TauPhi_fakes_stack_TauPt_liny`

Generated outputs or scripts:
- Script: `run/2017/validations/validate_low_met_fake_region.py`
- Output directory:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/plots/current_data_mc_fakes_stacks/
  ```
- Thesis image copies:
  ```text
  images/truth_and_fakes/medium_current/current_data_mc_fakes_stacks/
  ```

## Correction 3: Event Counts, Cutflows, And Pie Charts

Lines:
- Prong section text and table/figure block: `4666-4735`
- Charge section text and table/figure block: `4871-4942`

Original text anchor:
> "Relative contributions to signal from MC backgrounds are described..."

> "Full event counts and statistical uncertainties can be found in Table..."

Current status:
- The event-count tables, cutflows, and pie charts are stale.
- The tables quote old process groupings and old event yields.
- The current analysis split and fake bookkeeping are different from the old pie-chart logic:
  - the signal is the hadronic tau component of charged Drell--Yan production;
  - the leptonic tau component is treated as a simulated background contribution;
  - the jet-to-tau fake contribution is estimated with the data-driven fake-factor method.
- No current prong-split or charge-split event-count table matching the corrected final bookkeeping was found in the current output tree.

Reason:
- Keeping these tables and pie charts would imply that the old simulated background stack is still the final prediction.
- The current analysis uses a corrected pre-unfolding budget and a data-driven jet-fake estimate, so the old "Total MC" and pie-chart summaries are not the right summary of the final selected sample.

Literature and analysis evidence:
- The ATLAS high-mass $\tau+E_{\mathrm{T}}^{\mathrm{miss}}$ search separates data-driven jet backgrounds from backgrounds estimated with simulation: https://arxiv.org/abs/2402.16576.
- Current analysis evidence: `outputs/analysis_shadow_unfold/closure_summary.md`, section `Pre-unfolding budget`.
- Current budget for the nominal no-shadow configuration:
  - Data: `1351.000`
  - Simulated background contamination used in the corrected subtraction: `245.530`
  - Data-driven jet-to-tau fake estimate: `226.136`
  - Nonfiducial signal: `104.548`
  - Signal input after background subtraction: `774.786`

Suggested correction:
- Remove the stale prong and charge event-count/cutflow/pie-chart blocks unless they are regenerated with the current analysis bookkeeping.
- If a compact numerical summary is needed in Chapter 10, use a single current nominal-budget table rather than the old prong and charge tables.

Suggested text:

```tex
\begin{table}[h!]
    \centering
    \begin{tabular}{lr}
        Quantity & Yield \\
        \hline\hline
        Data & 1351.0 \\
        Simulated background contribution & 245.5 \\
        Data-driven jet-to-tau fake estimate & 226.1 \\
        Nonfiducial signal contribution & 104.5 \\
        Signal input after background subtraction & 774.8 \\
    \end{tabular}
    \caption{Nominal pre-unfolding event-yield budget for the Medium tau-identification signal region. The signal input is formed after subtracting the simulated background contribution, the data-driven jet-to-tau fake estimate, and the nonfiducial signal contribution from the selected data.}
    \label{tab:medium_current_preunfolding_budget}
\end{table}
```

Figures/tables affected:
- Remove or replace `tab:medium_event_counts_prongs`, lines `4670-4690`.
- Remove or replace `tab:medium_cutflow`, lines `4693-4721`.
- Remove or replace `fig:1prong_medium_signal_contribution_pie`, lines `4724-4728`.
- Remove or replace `fig:3prong_medium_signal_contribution_pie`, lines `4730-4734`.
- Remove or replace `tab:medium_event_counts_plusminus`, lines `4877-4897`.
- Remove or replace `tab:medium_cutflow_plusminus`, lines `4900-4928`.
- Remove or replace `fig:tauplus_medium_signal_contribution_pie`, lines `4931-4935`; note the current caption incorrectly says "loose ID".
- Remove or replace `fig:tauminus_medium_signal_contribution_pie`, lines `4937-4941`.

Generated outputs or scripts:
- Current numerical evidence: `outputs/analysis_shadow_unfold/closure_summary.md`.
- No current prong-split or charge-split replacement table was identified.

## Correction 4: Prong-Split Data/MC Comparisons

Lines:
- Discussion: `4666-4668`, `4737`
- Figure blocks: `4739-4825`

Original text anchor:
> "Comparisons between the full MC signal + background + fakes estimate and data..."

Current status:
- The prong-split comparison plots are stale `analysis_simple_2017` stack plots.
- They use old fake-estimate bookkeeping and old "full MC + fakes" wording.
- Current prong-split line-overlay replacements have now been generated for one-prong and three-prong tau candidates.
- The replacements use the same six observables as the old thesis blocks: $m_{\mathrm{T}}^W$, $E_{\mathrm{T}}^{\mathrm{miss}}$, tau transverse momentum, tau pseudorapidity, $|\Delta\phi(\tau,E_{\mathrm{T}}^{\mathrm{miss}})|$, and tau azimuthal angle.

Reason:
- Prong-split fake factors are central to the fake estimate, so retaining a prong-split validation section is now well-motivated.
- The regenerated plots use the current Medium-ID, low-$E_{\mathrm{T}}^{\mathrm{miss}}$, prong-split fake estimate and the same line-overlay style as the inclusive comparison.

Literature and analysis evidence:
- The ATLAS high-mass $\tau+E_{\mathrm{T}}^{\mathrm{miss}}$ search derives fake factors separately for one-prong and three-prong tau candidates, supporting prong-split validation when the plots are generated consistently: https://arxiv.org/abs/2402.16576.
- Current fake-factor validation: `outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md`.
- Current regenerated prong-split comparison plots:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/plots/current_data_mc_fakes_stacks/prong_split/
  ```
- Thesis image copies:
  ```text
  images/truth_and_fakes/medium_current/current_data_mc_fakes_stacks/prong_split/
  ```

Suggested correction:
- Replace the prong-split data/MC comparison figure blocks with the current line-overlay plots for one-prong and three-prong selections using the same three curves as the inclusive comparison:
  - data;
  - MC signal plus simulated backgrounds before adding fakes;
  - MC signal plus simulated backgrounds after adding the data-driven jet-to-tau fake estimate.
- Do not reuse the old `analysis_simple_2017/medium/1prong_/fakes/` or `analysis_simple_2017/medium/3prong_/fakes/` images.

Suggested text if the section is removed:

```tex
The fake-factor method is derived separately for one-prong and three-prong tau candidates, as described in Chapter~\ref{cha:fakes}. The inclusive Medium-ID comparison shown in Figure~\ref{fig:medium_current_data_mc_fakes} uses the combined prong-split fake estimate.
```

Figures/tables affected:
- Remove or replace `fig:stacks_with_fakes_1prong_medium`, lines `4739-4781`.
- Remove or replace `fig:stacks_with_fakes_3prong_medium`, lines `4783-4825`.

Generated outputs or scripts:
- Script: `run/2017/validations/validate_low_met_fake_region.py`.
- Current prong-split replacement plots have been generated and copied to the thesis image tree.

## Correction 5: Charge-Asymmetry Section

Lines:
- Text: `4871-4875`
- Figure blocks: `4944-5030`

Original text anchor:
> "Signal and background estimations between MC and data are also presented for charged positive and negative taus..."

Current status:
- The charge-asymmetry discussion is physically plausible in broad terms, but the current chapter still points to stale `analysis_simple_2017` images.
- Current charge-split data/MC replacement plots now exist and should be used instead.
- The charge-split tables and pie charts are stale.
- The section claims supplementary Loose and Tight comparisons remain available, but the current nominal analysis is Medium-only unless those supplementary plots are regenerated with the corrected fake estimate.
- Current charge-split line-overlay replacements have now been generated for positive and negative reconstructed tau candidates.

Reason:
- Charge-split comparisons can be useful because proton--proton collisions produce more $W^+$ than $W^-$ bosons, but this chapter should not show old plots as current evidence.
- Charge-split plots are now available using the same Medium-ID selection and data-driven jet-to-tau fake estimate as the inclusive comparison.
- This makes the charge-asymmetry section worth retaining as a validation cross-check, provided the stale tables and pie charts are removed or regenerated separately.

Literature and analysis evidence:
- ATLAS measurements of $W^\pm$ and $Z$ production at the LHC discuss the $W^+$ and $W^-$ production asymmetry in proton--proton collisions and its sensitivity to parton distribution functions: https://arxiv.org/abs/1603.09222.
- Current regenerated charge-split comparison plots:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/plots/current_data_mc_fakes_stacks/charge_split/
  ```
- Thesis image copies:
  ```text
  images/truth_and_fakes/medium_current/current_data_mc_fakes_stacks/charge_split/
  ```

Suggested correction:
- Retain the charge-asymmetry section as a validation cross-check, but replace the old charge-split images with the current line-overlay plots.
- Do not keep any quantitative statement about background fractions from the old tables unless those tables are regenerated with the current fake estimate.
- Do not keep old positive/negative tau plots under `analysis_simple_2017/medium/tauplus_/fakes/` or `analysis_simple_2017/medium/tauminus_/fakes/`.

Suggested text if the section is kept after regeneration:

```tex
The selected sample can also be separated by the charge of the reconstructed tau candidate. This provides a useful cross-check because $W^+$ and $W^-$ production rates differ in proton--proton collisions. The comparison is therefore performed separately for positive and negative tau candidates using the same Medium tau-identification selection and the same data-driven jet-to-tau fake estimate as in the inclusive comparison.
```

Figures/tables affected:
- Remove or replace `fig:stacks_with_fakes_tauplus_medium`, lines `4944-4986`.
- Remove or replace `fig:stacks_with_fakes_tauminus_medium`, lines `4988-5030`.
- Remove or replace the stale charge-yield table and cutflow listed in Correction 3.

Generated outputs or scripts:
- Script: `run/2017/validations/validate_low_met_fake_region.py`.
- Current charge-split replacement plots have been generated and copied to the thesis image tree.

## Correction 6: Wording And Labels For Fake Estimates

Lines:
- Opening paragraph: `4545`
- Inclusive figure captions: `4626`, `4632`, `4639`, `4645`, `4652`, `4658`, `4662`
- Prong figure captions: `4743`, `4749`, `4756`, `4762`, `4769`, `4775`, `4779`, `4787`, `4793`, `4800`, `4806`, `4813`, `4819`, `4823`
- Charge figure captions: `4948`, `4954`, `4961`, `4967`, `4974`, `4980`, `4984`, `4992`, `4998`, `5005`, `5011`, `5018`, `5024`, `5028`

Original text anchor:
> "fake jet background estimate"

> "full MC signal + background + fakes estimate"

Current status:
- The chapter uses several terms that are now inconsistent with Chapters 8 and 9.
- "Fake jet" should be avoided here because the background is a jet reconstructed as a tau candidate, not a fake jet.
- "Full MC signal + background + fakes" should be avoided because it can imply that the data-driven fake estimate is added on top of an all-MC fake-like background.

Reason:
- The corrected bookkeeping separates the data-driven jet-to-tau fake estimate from the simulated background contribution used in the selected signal-region comparison.
- This wording should align with ATLAS fake-factor terminology and with the rest of the thesis.

Literature and analysis evidence:
- ATLAS Universal Fake Factor paper uses "jets misidentified as tau leptons" and data-driven fake factors: https://arxiv.org/abs/2502.04156.
- ATLAS high-mass $\tau+E_{\mathrm{T}}^{\mathrm{miss}}$ search separates jet backgrounds estimated from data from other backgrounds estimated with simulation: https://arxiv.org/abs/2402.16576.

Suggested correction:
- Replace "fake jet background" with "jet-to-tau fake background" or "data-driven jet-to-tau fake estimate".
- Replace "full MC signal + background + fakes estimate" with "MC prediction before and after adding the data-driven jet-to-tau fake estimate".

Suggested text:

```tex
The MC prediction is shown before and after adding the data-driven jet-to-tau fake estimate.
```

Figures/tables affected:
- All Chapter 10 figure captions that refer to fake estimates.

Generated outputs or scripts:
- No regenerated output is required for this wording correction.

## Chapter 10 Figure And Table Checklist

This checklist is an inventory only; each correction above describes the required update.

- Lines `4578-4592`, `tab:triggers_2017`: remove from Chapter 10.
- Lines `4594-4620`, `tab:systematics_full`: remove from Chapter 10.
- Lines `4622-4664`, `fig:stacks_with_fakes_syst_medium`: replace with current inclusive Medium line-overlay plots under `truth_and_fakes/medium_current/current_data_mc_fakes_stacks/`.
- Lines `4670-4690`, `tab:medium_event_counts_prongs`: remove or regenerate with current analysis bookkeeping.
- Lines `4693-4721`, `tab:medium_cutflow`: remove or regenerate with current analysis bookkeeping.
- Lines `4724-4734`, prong pie charts: remove or regenerate with current analysis bookkeeping.
- Lines `4739-4781`, `fig:stacks_with_fakes_1prong_medium`: replace with current one-prong line-overlay plots under `truth_and_fakes/medium_current/current_data_mc_fakes_stacks/prong_split/`.
- Lines `4783-4825`, `fig:stacks_with_fakes_3prong_medium`: replace with current three-prong line-overlay plots under `truth_and_fakes/medium_current/current_data_mc_fakes_stacks/prong_split/`.
- Lines `4877-4897`, `tab:medium_event_counts_plusminus`: remove or regenerate with current analysis bookkeeping.
- Lines `4900-4928`, `tab:medium_cutflow_plusminus`: remove or regenerate with current analysis bookkeeping.
- Lines `4931-4941`, charge pie charts: remove or regenerate; line `4933` incorrectly says positive taus at Loose ID.
- Lines `4944-4986`, `fig:stacks_with_fakes_tauplus_medium`: replace with current positive-tau line-overlay plots under `truth_and_fakes/medium_current/current_data_mc_fakes_stacks/charge_split/`.
- Lines `4988-5030`, `fig:stacks_with_fakes_tauminus_medium`: replace with current negative-tau line-overlay plots under `truth_and_fakes/medium_current/current_data_mc_fakes_stacks/charge_split/`.

## Immediate To-Do List

1. Replace the Chapter 10 opening paragraph using Correction 1.
2. Remove the uncertainty equations and raw trigger/systematic tables from Chapter 10.
3. Replace the inclusive Medium figure block with the four current line-overlay plots listed in Correction 2.
4. Replace the stale prong-split and charge-split comparison blocks with the regenerated current Medium line-overlay plots.
5. Remove or regenerate the stale event-count, cutflow, and pie-chart summaries.
6. Standardise captions to use "data-driven jet-to-tau fake estimate" and avoid "full MC + fakes" wording.
7. After editing, compile only when ready to check figure placement and undefined references created by removing stale labels.
