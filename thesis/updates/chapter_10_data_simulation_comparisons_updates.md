# Chapter 10 Update Notes: Data-Simulation Comparisons

Source audited:
- Thesis source: `/mnt/B/Uni_Stuff_Queen_Mary/Documents/Thesis/thesis.tex`
- Chapter: `Data-Simulation Comparisons`
- Current chapter line range: `4619-5030`
- Relevant analysis scripts:
  - `run/2017/validations/validate_low_met_fake_region.py`
  - `run/2017/analysis_shadow_unfold.py`
  - `src/fakes.py`
- Relevant generated outputs:
  - `outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md`
  - `outputs/validate_shadow_fakes/low_met_fake_region/plots/chapter10_current_stacks/`
  - `outputs/validate_shadow_fakes/low_met_fake_region/tables/chapter10_current/`
  - `outputs/validate_shadow_fakes/low_met_fake_region/plots/chapter10_current_pies/`
  - `outputs/validate_shadow_fakes/low_met_fake_region/chapter10_systematics/root/`
- Current thesis image/table copies:
  - `images/truth_and_fakes/medium_current/chapter10_current_stacks/`
  - `images/truth_and_fakes/medium_current/chapter10_current_tables/`
  - `images/truth_and_fakes/medium_current/chapter10_current_pies/`

Purpose:
This note records what should be changed so Chapter 10 matches the current Medium tau-identification analysis, the current low-$E_{\mathrm{T}}^{\mathrm{miss}}$ data-driven jet-to-tau fake estimate, and the regenerated Chapter 10 validation figures.

## Executive Summary

Chapter 10 should now use the regenerated thesis-style stacked validation plots, not the older `analysis_simple_2017` plots and not the intermediate line-overlay plots. The current Chapter 10 assets are:

1. Medium tau-identification stack plots for the inclusive signal region, one-prong and three-prong selections, and positive- and negative-charge selections.
2. Stacked background components including the data-driven jet-to-tau fake estimate as a background component, with the hadronic $W\rightarrow\tau\nu$ signal shown as a line.
3. Grey uncertainty bands labelled `Stat. + Syst. Err.` in the plot legends. These include the plotted statistical uncertainty, the propagated statistical uncertainty on the fake-yield histogram, and the current MC systematic envelope available in the targeted Chapter 10 systematic cache. They do not include the dedicated fake-source shape envelopes for every displayed validation variable.
4. Regenerated Medium-ID event-yield tables for one-/three-prong and positive-/negative-charge selections.
5. Regenerated Medium-ID composition pie charts for the same prong and charge splits.

The main remaining edits are textual: the captions and opening paragraph should stop saying "fake jet estimation" and "full MC signal + background + fakes", and should instead describe the reconstructed prediction with the data-driven jet-to-tau fake estimate included as a background component.

## Correction 1: Opening Scope And Uncertainty Wording

Lines:
- `4621`

Original text anchor:
> "Signal and background estimations between MC and data are presented..."

> "Total uncertainties are displayed as discussed in Chapter~\ref{cha:uncertainties} including systematic and statistical uncertainties..."

Current status:
- The chapter is now correctly focused on the Medium tau-identification working point.
- The opening paragraph is too long and still uses imprecise wording.
- The phrase "total uncertainties" is too strong for the current Chapter 10 validation plots. The plotted band includes statistical uncertainty, the fake-yield statistical uncertainty, and the MC systematic envelope available in the Chapter 10 systematic cache, but not every dedicated fake-source shape uncertainty for every displayed validation variable.
- The phrase "fake jet background estimate" should be replaced by "jet-to-tau fake estimate" or "jet-to-tau fake background".

Reason:
- Chapter 10 is a validation chapter. It should describe the plotted reconstructed prediction and point the reader to Chapter 9 for the full uncertainty model.
- ATLAS fake-factor literature describes the relevant background as jets misidentified as hadronic tau candidates, not as "fake jets".
- The current plots are stack plots with process components, not line overlays.

Literature and analysis evidence:
- The ATLAS high-mass $\tau+E_{\mathrm{T}}^{\mathrm{miss}}$ search separates the jet background estimated from data from other backgrounds estimated using simulation: https://arxiv.org/abs/2402.16576.
- The ATLAS Universal Fake Factor paper motivates data-driven fake-factor estimates for jets misidentified as tau leptons: https://arxiv.org/abs/2502.04156.
- Analysis evidence: `outputs/validate_shadow_fakes/low_met_fake_region/low_met_fake_region_summary.md` reports `Chapter 10 systematic cache: available`.
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py:48-60` enables the Chapter 10 systematic cache, Chapter 10 current stack plots, and Chapter 10 current tables and pies.
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py:562-605` builds the `Stat. + Syst. Err.` band when systematic errors are available.

Suggested correction:
- Replace line `4621` with a shorter paragraph.
- Keep the chapter scoped to Medium-ID 2017 comparisons.
- Keep the uncertainty wording accurate: do not claim that the validation plots contain every fake-source shape uncertainty.

Suggested text:

```tex
Signal and background estimates are compared with the selected 2017 data in the Medium tau-identification signal region. The reconstructed prediction is shown as a stacked distribution containing the simulated background components and the data-driven jet-to-tau fake estimate, with the hadronic $W\rightarrow\tau\nu$ signal overlaid as a line. The comparison is shown for $\mtw$, $\etmiss$, $\taupt$, $\eta^\tau$, $|\Delta\phi(\tau,\etmiss)|$, and $\phi^\tau$. The grey bands show the statistical and MC systematic uncertainty components available for these validation distributions, while the full uncertainty model is discussed in Chapter~\ref{cha:uncertainties}. The same comparison is shown separately for one- and three-prong tau candidates in Figures~\cref{fig:stacks_with_fakes_1prong_medium,fig:stacks_with_fakes_3prong_medium}, and for positive and negative tau candidates in Figures~\cref{fig:stacks_with_fakes_tauplus_medium,fig:stacks_with_fakes_tauminus_medium}.
```

Figures/tables affected:
- All Chapter 10 figure captions should use wording consistent with this paragraph.

Generated outputs or scripts:
- Script: `run/2017/validations/validate_low_met_fake_region.py`
- Systematic cache:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/chapter10_systematics/root/
  ```
- Cached MC groups:
  ```text
  wtaunu_had.root
  wtaunu_lep.root
  wlnu.root
  zll.root
  top.root
  diboson.root
  ```

## Correction 2: Inclusive Medium Stack Figure Captions And Labels

Lines:
- Figure block: `4623-4663`
- Individual caption lines: `4627`, `4639`, `4645`, `4652`, `4658`
- Main caption: `4661`

Original text anchor:
> "data-MC comparison with full signal, background and $\taupt$-binned fake jet estimation..."

> "The grey band shows statistical uncertainty."

Current status:
- The image paths at lines `4626`, `4632`, `4638`, `4644`, `4651`, and `4657` already point to the current `chapter10_current_stacks/inclusive/` plot family.
- The captions are stale. They still refer to "fake jet estimation" and do not reflect the updated uncertainty band.
- The labels still contain old names ending in `TauPt_liny`; they are not fatal, but they are misleading because several plotted axes are logarithmic and the figure no longer represents a simple tau-transverse-momentum-binned line-overlay comparison.

Reason:
- The current inclusive plots are thesis-style stacked distributions. The data-driven jet-to-tau fake estimate is a stacked background component, and the signal is shown as a red line.
- Energy-binned variables use logarithmic x and y axes. The tau-pseudorapidity and tau-azimuth plots use the linear-y versions for readability.

Literature and analysis evidence:
- The ATLAS high-mass $\tau+E_{\mathrm{T}}^{\mathrm{miss}}$ search presents the hadronic tau plus missing-transverse-momentum final state using reconstructed $m_{\mathrm{T}}$ and missing-transverse-momentum observables: https://arxiv.org/abs/2402.16576.
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py:62-69` defines the six Chapter 10 variables.
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py:1802-1829` writes the inclusive Chapter 10 stack plots.
- Current thesis image copies:
  ```text
  images/truth_and_fakes/medium_current/chapter10_current_stacks/inclusive/
  ```

Suggested correction:
- Keep the current image paths.
- Replace the individual subfigure captions with short observable captions.
- Replace the main caption with a caption describing the stack composition and the uncertainty band.
- Optionally rename the subfigure labels to names that match the new figure family.

Suggested text:

```tex
\begin{figure}[h]
    \centering
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/chapter10_current_stacks/inclusive/medium_MTW_current_stack_fakes_log.png}
        \caption{$\mtw$.}
        \label{fig:medium_current_mtw_stack_fakes}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/chapter10_current_stacks/inclusive/medium_MET_met_current_stack_fakes_log.png}
        \caption{$\etmiss$.}
        \label{fig:medium_current_etmiss_stack_fakes}
    \end{subfigure}
    \vfill

    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/chapter10_current_stacks/inclusive/medium_TauPt_current_stack_fakes_log.png}
        \caption{$\taupt$.}
        \label{fig:medium_current_taupt_stack_fakes}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/chapter10_current_stacks/inclusive/medium_TauEta_current_stack_fakes_liny.png}
        \caption{$\eta^\tau$.}
        \label{fig:medium_current_taueta_stack_fakes}
    \end{subfigure}
    \hfill

    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/chapter10_current_stacks/inclusive/medium_AbsDeltaPhi_tau_met_current_stack_fakes_log.png}
        \caption{$|\Delta\phi(\tau,\etmiss)|$.}
        \label{fig:medium_current_dphitauetmiss_stack_fakes}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/chapter10_current_stacks/inclusive/medium_TauPhi_current_stack_fakes_liny.png}
        \caption{$\phi^\tau$.}
        \label{fig:medium_current_tauphi_stack_fakes}
    \end{subfigure}
    \caption{Comparison between 2017 data and the reconstructed prediction in the Medium tau-identification signal region. Simulated backgrounds and the data-driven jet-to-tau fake estimate are shown as stacked components, while the red line shows the hadronic $W\rightarrow\tau\nu$ signal. The lower panel shows the ratio of data to the total prediction. The grey band shows the statistical and MC systematic uncertainty components available for these validation distributions.}
    \label{fig:stacks_with_fakes_syst_medium}
\end{figure}
```

Figures/tables affected:
- `fig:stacks_with_fakes_syst_medium`, lines `4623-4663`.

Generated outputs or scripts:
- Script:
  ```bash
  VALIDATE_LOW_MET_BUILD_CHAPTER10_SYSTEMATICS=1 pixi run python run/2017/validations/validate_low_met_fake_region.py
  ```
- Output directory:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/plots/chapter10_current_stacks/inclusive/
  ```
- Copied thesis directory:
  ```text
  images/truth_and_fakes/medium_current/chapter10_current_stacks/inclusive/
  ```

## Correction 3: Prong-Split Tables, Pie Charts, And Cutflows

Lines:
- Prong section text: `4667`
- Event-count table: `4669-4690`
- Cutflow table: `4692-4720`
- Pie charts: `4723-4734`

Original text anchor:
> "Relative contributions to signal from MC backgrounds are described..."

> "Full event counts and statistical uncertainties can be found in Table~\ref{tab:medium_event_counts_prongs}..."

Current status:
- The old prong-split event-count table is stale and uses old process yields.
- A current replacement event-yield table has been generated and copied into the thesis image tree.
- The old cutflow table has not been regenerated in the current Chapter 10 output. It should be removed unless a new sequential cutflow is explicitly produced.
- The old pie charts under `analysis_simple_2017/` are stale. Current pie charts have been generated and copied.

Reason:
- The current event-yield table uses the corrected process split:
  - hadronic $W\rightarrow\tau\nu$ signal;
  - leptonic $W\rightarrow\tau\nu$ simulated background;
  - direct light-lepton charged Drell-Yan;
  - neutral Drell-Yan;
  - top;
  - diboson;
  - data-driven jet-to-tau fake estimate.
- The data-driven fake estimate should appear as a background component in the final selected-region yield summary.

Literature and analysis evidence:
- The ATLAS Universal Fake Factor paper motivates data-driven estimation of jets misidentified as tau leptons, with dependence on tau transverse momentum and charged-particle multiplicity: https://arxiv.org/abs/2502.04156.
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py:106-123` defines the Chapter 10 component labels.
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py:1941-1955` writes the current Chapter 10 yield tables and composition pies.
- Current output table:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/tables/chapter10_current/chapter10_medium_event_counts_prongs.tex
  ```
- Copied thesis table:
  ```text
  images/truth_and_fakes/medium_current/chapter10_current_tables/chapter10_medium_event_counts_prongs.tex
  ```
- Current pie charts:
  ```text
  images/truth_and_fakes/medium_current/chapter10_current_pies/prong_split/medium_1prong_current_prediction_pie.png
  images/truth_and_fakes/medium_current/chapter10_current_pies/prong_split/medium_3prong_current_prediction_pie.png
  ```

Suggested correction:
- Replace the old event-count table with an `\input`.
- Remove the old cutflow minipage unless a current cutflow is regenerated.
- Replace the old pie chart paths with the current pie chart paths.
- Update the surrounding text so it does not quote old background percentages.

Suggested text:

```tex
Relative contributions to the selected sample are shown separately for one- and three-prong tau candidates in Figures~\cref{fig:1prong_medium_signal_contribution_pie,fig:3prong_medium_signal_contribution_pie}. The corresponding weighted event yields are given in Table~\ref{tab:medium_event_counts_prongs}. The data-driven jet-to-tau fake estimate is included as a background contribution in both the table and the pie charts.

\begin{figure}[h]
    \centering
    \begin{minipage}{\textwidth}
        \centering
        \input{images/truth_and_fakes/medium_current/chapter10_current_tables/chapter10_medium_event_counts_prongs.tex}
        \captionof{table}{Weighted event counts for one- and three-prong tau candidates in the Medium tau-identification signal region. Uncertainties shown are statistical only.}
        \label{tab:medium_event_counts_prongs}
    \end{minipage}
    \vspace{5mm}

    \begin{minipage}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/chapter10_current_pies/prong_split/medium_1prong_current_prediction_pie.png}
        \captionof{figure}{Relative composition of the reconstructed prediction for one-prong tau candidates at Medium tau identification.}
        \label{fig:1prong_medium_signal_contribution_pie}
    \end{minipage}
    \hfill
    \begin{minipage}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/chapter10_current_pies/prong_split/medium_3prong_current_prediction_pie.png}
        \captionof{figure}{Relative composition of the reconstructed prediction for three-prong tau candidates at Medium tau identification.}
        \label{fig:3prong_medium_signal_contribution_pie}
    \end{minipage}
\end{figure}
```

Figures/tables affected:
- Replace `tab:medium_event_counts_prongs`, lines `4669-4690`.
- Remove `tab:medium_cutflow`, lines `4692-4720`, unless regenerated.
- Replace the pie-chart paths at lines `4724` and `4730`.

Generated outputs or scripts:
- Script: `run/2017/validations/validate_low_met_fake_region.py`
- Table output:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/tables/chapter10_current/chapter10_medium_event_counts_prongs.tex
  ```
- Pie outputs:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/plots/chapter10_current_pies/prong_split/
  ```

## Correction 4: Prong-Split Stack Figures

Lines:
- Discussion: `4736`
- One-prong figure: `4738-4780`
- Three-prong figure: `4782-4824`

Original text anchor:
> "Comparisons between the full MC signal + background + fakes estimate and data..."

Current status:
- Both prong-split figure blocks still point to stale `analysis_simple_2017` images.
- Current one-prong and three-prong thesis-style stack plots have been generated and copied into the thesis image tree.
- The current stack plots use the same visual convention as the inclusive figure: stacked simulated backgrounds plus data-driven jet-to-tau fakes, with the signal overlaid as a line and a `Stat. + Syst. Err.` band.

Reason:
- The fake-factor estimate is derived separately for one-prong and three-prong tau candidates, so retaining prong-split validation plots is useful.
- The old plots do not reflect the current Medium-ID, low-$E_{\mathrm{T}}^{\mathrm{miss}}$, prong-split fake-factor workflow.

Literature and analysis evidence:
- The ATLAS high-mass $\tau+E_{\mathrm{T}}^{\mathrm{miss}}$ search derives jet-background estimates with prong-dependent transfer factors, supporting prong-split validation: https://arxiv.org/abs/2402.16576.
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py:1831-1876` writes the one-prong and three-prong Chapter 10 stack plots.
- Current output directory:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/plots/chapter10_current_stacks/prong_split/
  ```
- Copied thesis directory:
  ```text
  images/truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/
  ```

Suggested correction:
- Replace every `analysis_simple_2017/medium/1prong_/fakes/...` path with the corresponding current path under `truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/1prong/`.
- Replace every `analysis_simple_2017/medium/3prong_/fakes/...` path with the corresponding current path under `truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/3prong/`.
- Update captions to remove "full MC signal + background + fakes estimate".

Suggested path replacements:

```text
analysis_simple_2017/medium/1prong_/fakes/1prong_medium_MTW_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/1prong/medium_1prong_MTW_current_stack_fakes_log.png

analysis_simple_2017/medium/1prong_/fakes/1prong_medium_MET_met_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/1prong/medium_1prong_MET_met_current_stack_fakes_log.png

analysis_simple_2017/medium/1prong_/fakes/1prong_medium_TauPt_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/1prong/medium_1prong_TauPt_current_stack_fakes_log.png

analysis_simple_2017/medium/1prong_/fakes/1prong_medium_TauEta_stack_fakes_liny.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/1prong/medium_1prong_TauEta_current_stack_fakes_liny.png

analysis_simple_2017/medium/1prong_/fakes/1prong_medium_AbsDeltaPhi_tau_met_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/1prong/medium_1prong_AbsDeltaPhi_tau_met_current_stack_fakes_log.png

analysis_simple_2017/medium/1prong_/fakes/1prong_medium_TauPhi_stack_fakes_liny.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/1prong/medium_1prong_TauPhi_current_stack_fakes_liny.png

analysis_simple_2017/medium/3prong_/fakes/3prong_medium_MTW_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/3prong/medium_3prong_MTW_current_stack_fakes_log.png

analysis_simple_2017/medium/3prong_/fakes/3prong_medium_MET_met_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/3prong/medium_3prong_MET_met_current_stack_fakes_log.png

analysis_simple_2017/medium/3prong_/fakes/3prong_medium_TauPt_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/3prong/medium_3prong_TauPt_current_stack_fakes_log.png

analysis_simple_2017/medium/3prong_/fakes/3prong_medium_TauEta_stack_fakes_liny.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/3prong/medium_3prong_TauEta_current_stack_fakes_liny.png

analysis_simple_2017/medium/3prong_/fakes/3prong_medium_AbsDeltaPhi_tau_met_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/3prong/medium_3prong_AbsDeltaPhi_tau_met_current_stack_fakes_log.png

analysis_simple_2017/medium/3prong_/fakes/3prong_medium_TauPhi_stack_fakes_liny.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/prong_split/3prong/medium_3prong_TauPhi_current_stack_fakes_liny.png
```

Suggested caption replacement:

```tex
\caption{Comparison between 2017 data and the reconstructed prediction for one-prong tau candidates in the Medium tau-identification signal region. Simulated backgrounds and the data-driven jet-to-tau fake estimate are shown as stacked components, while the red line shows the hadronic $W\rightarrow\tau\nu$ signal. The lower panel shows the ratio of data to the total prediction. The grey band shows the statistical and MC systematic uncertainty components available for these validation distributions.}
```

Use the same wording for the three-prong figure, replacing "one-prong" with "three-prong".

Figures/tables affected:
- Replace all paths and captions in `fig:stacks_with_fakes_1prong_medium`, lines `4738-4780`.
- Replace all paths and captions in `fig:stacks_with_fakes_3prong_medium`, lines `4782-4824`.

Generated outputs or scripts:
- Script:
  ```bash
  VALIDATE_LOW_MET_BUILD_CHAPTER10_SYSTEMATICS=1 pixi run python run/2017/validations/validate_low_met_fake_region.py
  ```
- Output directory:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/plots/chapter10_current_stacks/prong_split/
  ```

## Correction 5: Charge-Split Tables, Pie Charts, And Text

Lines:
- Charge section text: `4872-4874`
- Event-count table: `4876-4897`
- Cutflow table: `4899-4927`
- Pie charts: `4930-4941`

Original text anchor:
> "As the cross-section for $W^+\rightarrow \ell\nu$ is expected to be higher..."

> "Reconstructed (MC) relative contributions to the signal region for positive taus at loose ID."

Current status:
- The charge-asymmetry discussion is useful, but the quoted background fractions are from stale tables.
- The old charge event-count table is stale.
- A current charge-split event-yield table has been generated and copied into the thesis image tree.
- The old cutflow table has not been regenerated in the current Chapter 10 output and should be removed unless a new sequential cutflow is explicitly produced.
- The old positive-tau pie caption incorrectly says "loose ID"; it should be replaced.
- Current charge-split pie charts have been generated and copied.

Reason:
- Positive- and negative-charge comparisons are a useful validation cross-check because $W^+$ and $W^-$ production rates differ in proton-proton collisions.
- The current table and pie charts include the corrected data-driven jet-to-tau fake estimate as a background component.

Literature and analysis evidence:
- ATLAS measurements of $W^\pm$ and $Z$ production at $\sqrt{s}=13$ TeV discuss the different $W^+$ and $W^-$ production rates in proton-proton collisions and their sensitivity to parton distribution functions: https://arxiv.org/abs/1603.09222.
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py:1878-1939` writes the charge-split Chapter 10 stack plots.
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py:1941-1955` writes the charge-split yield table and pie charts.
- Current output table:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/tables/chapter10_current/chapter10_medium_event_counts_plusminus.tex
  ```
- Copied thesis table:
  ```text
  images/truth_and_fakes/medium_current/chapter10_current_tables/chapter10_medium_event_counts_plusminus.tex
  ```
- Current pie charts:
  ```text
  images/truth_and_fakes/medium_current/chapter10_current_pies/charge_split/medium_tauplus_current_prediction_pie.png
  images/truth_and_fakes/medium_current/chapter10_current_pies/charge_split/medium_tauminus_current_prediction_pie.png
  ```

Suggested correction:
- Replace the text at lines `4872-4874` with a shorter charge-validation paragraph.
- Replace the event-count table with an `\input`.
- Remove the old cutflow minipage unless a current cutflow is regenerated.
- Replace the old pie chart paths and captions.

Suggested text:

```tex
The selected sample is also separated by the charge of the reconstructed tau candidate. This provides a useful validation cross-check because $W^+$ and $W^-$ production rates differ in proton--proton collisions. The same Medium tau-identification selection and data-driven jet-to-tau fake estimate are used for the positive- and negative-tau comparisons.

\begin{figure}[h]
    \centering
    \begin{minipage}{\textwidth}
        \centering
        \input{images/truth_and_fakes/medium_current/chapter10_current_tables/chapter10_medium_event_counts_plusminus.tex}
        \captionof{table}{Weighted event counts for positive and negative tau candidates in the Medium tau-identification signal region. Uncertainties shown are statistical only.}
        \label{tab:medium_event_counts_plusminus}
    \end{minipage}
    \vspace{5mm}

    \begin{minipage}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/chapter10_current_pies/charge_split/medium_tauplus_current_prediction_pie.png}
        \captionof{figure}{Relative composition of the reconstructed prediction for positive tau candidates at Medium tau identification.}
        \label{fig:tauplus_medium_signal_contribution_pie}
    \end{minipage}
    \hfill
    \begin{minipage}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{truth_and_fakes/medium_current/chapter10_current_pies/charge_split/medium_tauminus_current_prediction_pie.png}
        \captionof{figure}{Relative composition of the reconstructed prediction for negative tau candidates at Medium tau identification.}
        \label{fig:tauminus_medium_signal_contribution_pie}
    \end{minipage}
\end{figure}
```

Figures/tables affected:
- Replace `tab:medium_event_counts_plusminus`, lines `4876-4897`.
- Remove `tab:medium_cutflow_plusminus`, lines `4899-4927`, unless regenerated.
- Replace the pie chart paths at lines `4931` and `4937`.
- Correct the positive-tau pie caption at line `4932`.

Generated outputs or scripts:
- Script: `run/2017/validations/validate_low_met_fake_region.py`
- Table output:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/tables/chapter10_current/chapter10_medium_event_counts_plusminus.tex
  ```
- Pie outputs:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/plots/chapter10_current_pies/charge_split/
  ```

## Correction 6: Charge-Split Stack Figures

Lines:
- Positive-tau figure: `4943-4985`
- Negative-tau figure: `4987-5029`

Original text anchor:
> "Comparisons between the full MC signal, background, and fakes estimate and data..."

Current status:
- Both charge-split figure blocks still point to stale `analysis_simple_2017` images.
- Current positive- and negative-tau thesis-style stack plots have been generated and copied into the thesis image tree.
- One subfigure label at line `4998` is duplicated or misleading: `fig:medium_MET_met_stack_fakes_liny` is not charge-specific.

Reason:
- Charge-split validation is useful, but only if the plots use the current Medium-ID selection and the current fake estimate.
- The current plots use the same style and uncertainty-band treatment as the inclusive and prong-split figures.

Literature and analysis evidence:
- ATLAS $W^\pm$ and $Z$ production measurements motivate charge-separated checks in proton-proton data: https://arxiv.org/abs/1603.09222.
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py:1878-1939` writes the charge-split plots.
- Current output directory:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/plots/chapter10_current_stacks/charge_split/
  ```
- Copied thesis directory:
  ```text
  images/truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/
  ```

Suggested correction:
- Replace every `analysis_simple_2017/medium/tauplus_/fakes/...` path with the corresponding current path under `truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauplus/`.
- Replace every `analysis_simple_2017/medium/tauminus_/fakes/...` path with the corresponding current path under `truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauminus/`.
- Update captions to remove "full MC signal + background + fakes estimate".
- Make all subfigure labels charge-specific.

Suggested path replacements:

```text
analysis_simple_2017/medium/tauplus_/fakes/tauplus_medium_MTW_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauplus/medium_tauplus_MTW_current_stack_fakes_log.png

analysis_simple_2017/medium/tauplus_/fakes/tauplus_medium_MET_met_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauplus/medium_tauplus_MET_met_current_stack_fakes_log.png

analysis_simple_2017/medium/tauplus_/fakes/tauplus_medium_TauPt_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauplus/medium_tauplus_TauPt_current_stack_fakes_log.png

analysis_simple_2017/medium/tauplus_/fakes/tauplus_medium_TauEta_stack_fakes_liny.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauplus/medium_tauplus_TauEta_current_stack_fakes_liny.png

analysis_simple_2017/medium/tauplus_/fakes/tauplus_medium_AbsDeltaPhi_tau_met_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauplus/medium_tauplus_AbsDeltaPhi_tau_met_current_stack_fakes_log.png

analysis_simple_2017/medium/tauplus_/fakes/tauplus_medium_TauPhi_stack_fakes_liny.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauplus/medium_tauplus_TauPhi_current_stack_fakes_liny.png

analysis_simple_2017/medium/tauminus_/fakes/tauminus_medium_MTW_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauminus/medium_tauminus_MTW_current_stack_fakes_log.png

analysis_simple_2017/medium/tauminus_/fakes/tauminus_medium_MET_met_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauminus/medium_tauminus_MET_met_current_stack_fakes_log.png

analysis_simple_2017/medium/tauminus_/fakes/tauminus_medium_TauPt_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauminus/medium_tauminus_TauPt_current_stack_fakes_log.png

analysis_simple_2017/medium/tauminus_/fakes/tauminus_medium_TauEta_stack_fakes_liny.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauminus/medium_tauminus_TauEta_current_stack_fakes_liny.png

analysis_simple_2017/medium/tauminus_/fakes/tauminus_medium_AbsDeltaPhi_tau_met_stack_fakes_log.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauminus/medium_tauminus_AbsDeltaPhi_tau_met_current_stack_fakes_log.png

analysis_simple_2017/medium/tauminus_/fakes/tauminus_medium_TauPhi_stack_fakes_liny.png
-> truth_and_fakes/medium_current/chapter10_current_stacks/charge_split/tauminus/medium_tauminus_TauPhi_current_stack_fakes_liny.png
```

Suggested caption replacement:

```tex
\caption{Comparison between 2017 data and the reconstructed prediction for positive tau candidates in the Medium tau-identification signal region. Simulated backgrounds and the data-driven jet-to-tau fake estimate are shown as stacked components, while the red line shows the hadronic $W\rightarrow\tau\nu$ signal. The lower panel shows the ratio of data to the total prediction. The grey band shows the statistical and MC systematic uncertainty components available for these validation distributions.}
```

Use the same wording for the negative-tau figure, replacing "positive" with "negative".

Figures/tables affected:
- Replace all paths and captions in `fig:stacks_with_fakes_tauplus_medium`, lines `4943-4985`.
- Replace all paths and captions in `fig:stacks_with_fakes_tauminus_medium`, lines `4987-5029`.

Generated outputs or scripts:
- Script:
  ```bash
  VALIDATE_LOW_MET_BUILD_CHAPTER10_SYSTEMATICS=1 pixi run python run/2017/validations/validate_low_met_fake_region.py
  ```
- Output directory:
  ```text
  outputs/validate_shadow_fakes/low_met_fake_region/plots/chapter10_current_stacks/charge_split/
  ```

## Correction 7: Plot Legend Terminology

Lines:
- All current stack plots used in Chapter 10.

Current status:
- The copied plots are usable and up to date.
- The plot legend currently labels the data-driven fake component as `Fake Jets`.
- The generated tables use the clearer label `Jet-to-tau fakes`.

Reason:
- The literature and the rest of the thesis wording refer to jets misidentified as tau candidates, or to a jet-to-tau fake estimate. "Fake Jets" can be read as jets being fake, rather than jets faking tau candidates.

Literature and analysis evidence:
- ATLAS Universal Fake Factor paper: https://arxiv.org/abs/2502.04156.
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py:113` labels table rows as `Jet-to-tau fakes`.
- Analysis evidence: `run/2017/validations/validate_low_met_fake_region.py:579` labels the stack legend as `Fake Jets`.

Suggested correction:
- The thesis captions should use "data-driven jet-to-tau fake estimate" even if the current plot legend says `Fake Jets`.
- For the cleanest final thesis version, regenerate the Chapter 10 plots after changing the legend label from `Fake Jets` to `Jet-to-tau fakes`.

Suggested text:

```tex
The data-driven jet-to-tau fake estimate is included as a background component.
```

Figures/tables affected:
- All Chapter 10 stack plots.

Generated outputs or scripts:
- Current plots are already generated and copied.
- Optional follow-up script edit: change `stack_labels = ... + ["Fake Jets"]` to use `Jet-to-tau fakes` in `run/2017/validations/validate_low_met_fake_region.py`.

## Chapter 10 Figure And Table Checklist

This checklist is an inventory only; each correction above explains what to do.

- Line `4621`: replace the opening paragraph; avoid "total uncertainties" unless the full fake-source shape envelope is included in every displayed validation variable.
- Lines `4623-4663`, `fig:stacks_with_fakes_syst_medium`: keep the current `chapter10_current_stacks/inclusive/` image paths, but update captions and labels.
- Lines `4669-4690`, `tab:medium_event_counts_prongs`: replace with `images/truth_and_fakes/medium_current/chapter10_current_tables/chapter10_medium_event_counts_prongs.tex`.
- Lines `4692-4720`, `tab:medium_cutflow`: remove unless a current cutflow table is regenerated.
- Lines `4723-4734`, prong pie charts: replace with `chapter10_current_pies/prong_split/medium_1prong_current_prediction_pie.png` and `medium_3prong_current_prediction_pie.png`.
- Lines `4738-4780`, `fig:stacks_with_fakes_1prong_medium`: replace all stale `analysis_simple_2017` paths with `chapter10_current_stacks/prong_split/1prong/` paths.
- Lines `4782-4824`, `fig:stacks_with_fakes_3prong_medium`: replace all stale `analysis_simple_2017` paths with `chapter10_current_stacks/prong_split/3prong/` paths.
- Lines `4872-4874`: replace the charge-asymmetry text; remove stale numerical background-fraction claims.
- Lines `4876-4897`, `tab:medium_event_counts_plusminus`: replace with `images/truth_and_fakes/medium_current/chapter10_current_tables/chapter10_medium_event_counts_plusminus.tex`.
- Lines `4899-4927`, `tab:medium_cutflow_plusminus`: remove unless a current cutflow table is regenerated.
- Lines `4930-4941`, charge pie charts: replace with `chapter10_current_pies/charge_split/medium_tauplus_current_prediction_pie.png` and `medium_tauminus_current_prediction_pie.png`.
- Lines `4943-4985`, `fig:stacks_with_fakes_tauplus_medium`: replace all stale `analysis_simple_2017` paths with `chapter10_current_stacks/charge_split/tauplus/` paths.
- Lines `4987-5029`, `fig:stacks_with_fakes_tauminus_medium`: replace all stale `analysis_simple_2017` paths with `chapter10_current_stacks/charge_split/tauminus/` paths.

## Immediate To-Do List

1. Replace the Chapter 10 opening paragraph using Correction 1.
2. Update the inclusive figure captions using Correction 2.
3. Replace the prong-split event-count table and pie charts using Correction 3; remove the stale prong cutflow unless a current cutflow is regenerated.
4. Replace all prong-split stack plot paths and captions using Correction 4.
5. Replace the charge-split text, event-count table, and pie charts using Correction 5; remove the stale charge cutflow unless a current cutflow is regenerated.
6. Replace all charge-split stack plot paths and captions using Correction 6.
7. Optionally regenerate the Chapter 10 stack plots once more with the legend label changed from `Fake Jets` to `Jet-to-tau fakes`.
8. After editing the thesis, compile and check for undefined labels caused by removing the old cutflow tables.
