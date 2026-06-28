# Chapter 9 Update Notes: Uncertainties

Source audited:
- Thesis source: `/mnt/B/Uni_Stuff_Queen_Mary/Documents/Thesis/thesis.tex`
- Chapter: `Uncertainties`
- Current chapter line range: `4413-4614`
- Relevant analysis scripts:
  - `run/2017/analysis_shadow_unfold.py`
  - `run/2017/shadow_unfold/systematics.py`
  - `src/dataset.py`
  - `src/unfolding.py`
- Relevant generated outputs:
  - `outputs/analysis_shadow_unfold/closure_summary.md`
  - `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/systematics/no_shadow_bin_MTW_response_systematics.png`
  - `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/systematics/MTW_shadow_bin_250_MTW_response_systematics.png`
  - `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/fake_source_systematics/no_shadow_bin_MTW_4iter_combined_fake_source_uncertainty.png`
  - `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/fake_source_systematics/MTW_shadow_bin_250_MTW_4iter_combined_fake_source_uncertainty.png`
  - `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/fake_width_systematic/no_shadow_bin_MTW_TauTrackWidthPt1000PV_4iter_fake_width_uncertainty.png`
  - `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/fake_width_systematic/MTW_shadow_bin_250_MTW_TauTrackWidthPt1000PV_4iter_fake_width_uncertainty.png`

Purpose:
This note records the remaining Chapter 9 changes needed after the recent partial update. It is scoped only to Chapter 9 and uses both line numbers and original-text anchors, because line numbers will shift as edits are made.

## Executive Summary

Chapter 9 has been partially updated and is closer to the current analysis than the previous draft. The statistical-uncertainty section is mostly usable. The remaining issues are:

1. Clean the opening scope and experimental-uncertainty wording so it does not expose raw implementation names.
2. Replace the old TES and tau-efficiency figure blocks with current response-level uncertainty material.
3. Finish the tau-efficiency rewrite by removing duplicated old prose, stale figures, and raw trigger/systematic component names from paste-ready text.
4. Tighten the jet-to-tau fake uncertainty section so it states the current three components: fake-factor statistical uncertainty, low-$E_{\mathrm{T}}^{\mathrm{miss}}$ control-region transfer, and the validated 1-prong tau-width composition variation.
5. Add a short method-uncertainty section for unfolding closure, iteration choice, and response-configuration dependence.
6. Fix duplicate LaTeX labels and the remaining old-style inline maths in Chapter 9.

## Correction 1: Opening Scope And Experimental-Uncertainty Vocabulary

Lines:
- `4422`
- `4438`

Original text anchor:
> "Experimental uncertainties are represented in `\lstinline{up}' and `\lstinline{down}' variations..."

> "Experimental variations on tau reconstruction and identification are considered..."

Current status:
- The current opening correctly mentions the reconstructed tau response, data-driven jet-fake estimate, statistics, and unfolding procedure.
- The phrase using `\lstinline{up}` and `\lstinline{down}` is implementation-facing and should be replaced by ordinary thesis prose.
- Line `4438` says experimental uncertainties are split into two groups, but the current chapter now discusses tau energy scale, tau-object efficiency, and fake-source uncertainties. The opening of the experimental section should therefore be more precise.

Reason:
- The current analysis propagates detector-response variations through the response matrix and handles jet-to-tau fake uncertainties through fake-factor variations.
- Chapter 9 should describe the uncertainty model in physics terms, not systematic-branch suffixes.

Literature and analysis evidence:
- RooUnfold describes unfolded covariance/error propagation and response-based treatment of detector effects: https://arxiv.org/abs/1105.1160.
- ATLAS tau performance literature treats tau reconstruction, identification, trigger efficiency, and energy calibration as tau-object performance ingredients: https://arxiv.org/abs/1412.7086.
- Analysis evidence: `outputs/analysis_shadow_unfold/closure_summary.md` records full fake-source systematics and closure checks.

Suggested correction:
- Replace line `4422` with prose using upward/downward $1\sigma$ variations.
- Replace line `4438` with a short scope sentence for tau energy-scale and tau-object efficiency uncertainties.

Suggested text:

```tex
In this analysis, the dominant uncertainty sources are expected to come from the reconstructed hadronic-tau response, the data-driven jet-fake estimate, the limited event statistics, and the unfolding procedure. Experimental uncertainties are evaluated using upward and downward variations corresponding to $+1\sigma$ and $-1\sigma$ shifts from the nominal response. Uncertainties associated with the jet-to-tau fake background are treated separately, using variations of the fake-factor method described in Chapter~\ref{cha:fakes}. Method uncertainties are assessed through closure tests and variations of the unfolding configuration.
```

Suggested text:

```tex
Experimental uncertainties in this analysis are dominated by the hadronic-tau response. The two main classes considered here are the tau energy scale and the tau-object efficiency uncertainties, both of which affect the reconstructed-to-fiducial response used in the unfolding.
```

Figures/tables affected:
- None directly.

Generated outputs or scripts:
- `outputs/analysis_shadow_unfold/closure_summary.md`

## Correction 2: Tau Energy-Scale Prose And Figure Block

Lines:
- Prose: `4442-4450`
- Figure block: `4453-4490`

Original text anchor:
> "These are provided by the \lstinline{Athena} analysis tooling as separate, simulation samples..."

> `systematic_alt_binning/medium/systematics/wtaunu/...`

Current status:
- The physics topic is correct: tau energy scale is an important experimental uncertainty.
- The wording is still too implementation-facing and partly inaccurate:
  - TES variations shift reconstructed kinematics; they are not just ordinary event-weight variations.
  - The chapter should not describe the software layer that provides the variations.
  - The old plots are from `2024-12-20` and do not reflect the current shadow-unfolding response-systematic treatment.
- The old figure block includes tau transverse-momentum and tau pseudorapidity plots, while the current final uncertainty propagation is for the unfolded $m_{\mathrm{T}}^W$ spectrum.

Reason:
- The current analysis rebuilds or reuses shifted response objects and unfolds with the shifted response. The uncertainty is defined from the change in the unfolded $m_{\mathrm{T}}^W$ spectrum relative to nominal.
- Current response-systematic plots were generated on `2026-06-24`.

Literature and analysis evidence:
- ATLAS tau performance documentation describes tau energy calibration and its uncertainty for hadronically decaying tau candidates: https://arxiv.org/abs/1412.7086.
- RooUnfold supports propagation of detector response effects through unfolded spectra: https://arxiv.org/abs/1105.1160.
- Current generated plots:
  - `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/systematics/no_shadow_bin_MTW_response_systematics.png`
  - `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/systematics/MTW_shadow_bin_250_MTW_response_systematics.png`

Suggested correction:
- Replace lines `4442-4450` with response-level TES wording.
- Replace the six-panel old TES figure block with a two-panel current response-systematic figure.
- Before compiling, copy the current outputs into the thesis image tree, for example:
  - `images/uncertainties/current/no_shadow_bin_MTW_response_systematics.png`
  - `images/uncertainties/current/MTW_shadow_bin_250_MTW_response_systematics.png`

Suggested text:

```tex
Tau energy-scale uncertainties account for imperfect knowledge of the visible energy calibration of hadronically decaying tau candidates. These uncertainties include detector-response, in-situ calibration, modelling, and hadronic-shower components, and are evaluated through upward and downward variations of the reconstructed tau kinematics. Since the unfolded $\mtw$ spectrum depends on both the reconstructed tau momentum and the missing transverse momentum, the tau energy-scale variations are propagated through the reconstructed-to-fiducial response matrix rather than treated only as a reconstructed-yield uncertainty.

For each variation, the response matrix is rebuilt and the measured spectrum is unfolded with the shifted response. The resulting change in the unfolded spectrum relative to the nominal result is taken as the tau energy-scale contribution to the response uncertainty.
```

Suggested figure replacement:

```tex
\begin{figure}
    \centering
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{uncertainties/current/no_shadow_bin_MTW_response_systematics.png}
        \caption{Response-systematic uncertainty on the unfolded $\mtw$ spectrum for the nominal no-shadow response.}
        \label{fig:no_shadow_bin_MTW_response_systematics}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{uncertainties/current/MTW_shadow_bin_250_MTW_response_systematics.png}
        \caption{Response-systematic uncertainty on the unfolded $\mtw$ spectrum for the $\mtw$-shadow response.}
        \label{fig:MTW_shadow_bin_250_MTW_response_systematics}
    \end{subfigure}
    \caption{Experimental response uncertainties propagated through the unfolding response for the two response configurations used in the current analysis.}
    \label{fig:response_systematics_current}
\end{figure}
```

Figures/tables affected:
- Replace lines `4453-4490`.
- Old figure paths under `systematic_alt_binning/medium/systematics/wtaunu/` should not be presented as the current uncertainty result.

Generated outputs or scripts:
- Current output plots are under `outputs/analysis_shadow_unfold/plots/.../systematics/`.
- Main production script: `run/2017/analysis_shadow_unfold.py`.
- TES response helper: `run/2017/shadow_unfold/systematics.py`.

## Correction 3: Tau Efficiency And Trigger Subsections

Lines:
- `4492-4602`

Original text anchor:
> "These are provided under the \lstinline{Athena} framework as \lstinline{RECO_TOTAL} uncertainties..."

> "These are formed of three components: \lstinline{TRIGGER_STATDATA}, \lstinline{TRIGGER_STATMC}, and \lstinline{TRIGGER_SYST}..."

> "Tau identification efficiency uncertainties ... are modelled by tag-and-probe analyses in MC samples..."

Current status:
- The new leading paragraph at lines `4492-4500` is directionally correct.
- Line `4502` repeats old reconstruction-efficiency prose and reintroduces raw implementation names.
- The reconstruction and trigger figure blocks at lines `4504-4595` are stale `analysis_simple_2017` plots from `2025-04-13`.
- Trigger wording still mixes old unsupported percentage claims with the updated prose.
- The identification paragraph is improved, but it should avoid implying that tau-ID calibrations are only MC tag-and-probe and should more clearly separate true-tau ID efficiency from jet-to-tau fake modelling.

Reason:
- The current response cache includes tau reconstruction, overlap-removal, and trigger efficiency variation families for simulated genuine hadronic tau candidates.
- The current final response-systematic treatment does not use ordinary true-tau ID efficiency variations to model jets misidentified as tau candidates; jet-to-tau misidentification is handled by the data-driven fake-factor estimate.

Literature and analysis evidence:
- ATLAS tau performance documentation covers trigger, reconstruction, identification, and energy calibration algorithms for hadronic tau decays: https://arxiv.org/abs/1412.7086.
- The high-mass $\tau+E_{\mathrm{T}}^{\mathrm{miss}}$ search separates jet backgrounds estimated from data from other simulated backgrounds, supporting the distinction between genuine-tau object efficiencies and jet-to-tau fake modelling: https://arxiv.org/abs/2402.16576.
- Analysis evidence:
  - `src/dataset.py:163-169` collects tau efficiency weight variations.
  - `run/2017/analysis_shadow_unfold.py:127-130` excludes raw true-hadronic-tau RNN/jet-ID efficiency families from the response systematic set.

Suggested correction:
- Replace the whole tau-efficiency section from line `4492` through line `4602`.
- Do not keep the old reconstruction and trigger figure blocks unless regenerated from the current response-systematic workflow.

Suggested text:

```tex
\subsection{Tau Efficiency Uncertainties}

Tau efficiency uncertainties account for residual differences between data and MC simulation in the probability for a genuine $\tau_{\mathrm{had-vis}}$ candidate to be reconstructed, identified, and selected by the trigger. These effects are applied as efficiency scale-factor variations for simulated genuine hadronic tau candidates and are propagated through the signal response. They therefore affect both the reconstructed signal yield and the migration between reconstructed and fiducial bins in selected analysis variables. The treatment follows the standard ATLAS separation of tau reconstruction, identification, trigger, and energy-calibration performance uncertainties~\cite{tau_recon_2015,tau_recon_2016}.

\subsubsection{Reconstruction Efficiencies}

Tau reconstruction efficiency uncertainties describe the uncertainty in the probability for a genuine visible hadronic tau decay to be reconstructed as a $\tau_{\mathrm{had-vis}}$ candidate. These uncertainties include residual differences between data and simulation in the reconstruction efficiency, as well as effects that become important for high-$\pt$ tau candidates. For three-prong tau decays, an additional high-$\pt$ uncertainty accounts for possible mis-modelling of the probability for nearby charged-particle tracks to merge or be reconstructed incorrectly. In this analysis, the reconstruction-efficiency variations are propagated through the selected $W\rightarrow\tau\nu\rightarrow\mathrm{hadrons}$ signal sample.

Efficiency effects associated with the overlap-removal procedure are treated in the same class of object-level uncertainties. In particular, the electron-overlap-removal efficiency accounts for differences between data and simulation in the probability that a genuine hadronic tau candidate is retained after rejecting candidates overlapping with reconstructed electrons.

\subsubsection{Trigger Efficiencies}

Tau trigger efficiency uncertainties describe the uncertainty in the probability for a selected $\tau_{\mathrm{had-vis}}$ candidate to satisfy the tau-trigger requirement. They include the statistical uncertainty of the efficiency measurement in data, the statistical uncertainty from the simulated samples used in the calibration, and the systematic uncertainty associated with the trigger-efficiency measurement. These components are propagated as separate variations and are combined with the other experimental uncertainties in the final response uncertainty.

The trigger-efficiency uncertainty is relevant for this measurement because the selected phase space contains high-$\pt$ hadronic tau candidates and large missing transverse momentum. Although the analysis is above the main trigger turn-on region, the residual uncertainty in the tau-trigger efficiency is retained and propagated to the reconstructed $\mtw$ spectrum.

\subsubsection{Identification Efficiencies}

Tau identification efficiency uncertainties describe the uncertainty in the probability for a genuine $\tau_{\mathrm{had-vis}}$ candidate to pass the tau-identification working point. In this analysis the nominal selected tau candidate is required to satisfy the Medium tau-identification requirement. The corresponding efficiency uncertainty is conceptually distinct from the probability for a quark- or gluon-initiated jet to be misidentified as a hadronic tau candidate.

For this reason, tau-identification efficiency uncertainties for genuine hadronic tau candidates are not used to model the jet-to-tau fake background. The jet-to-tau contribution is estimated with the data-driven fake-factor method described in Section~\ref{sec:fakes_uncert}, with its own uncertainty treatment. This avoids mixing the calibration of genuine $\tau_{\mathrm{had-vis}}$ candidates with the modelling of jets that pass the tau-identification requirement.
```

Figures/tables affected:
- Remove lines `4504-4545`, the old reconstruction-efficiency figure block.
- Remove lines `4553-4595`, the old trigger-efficiency figure block.
- These figure blocks also contain duplicate labels listed in Correction 6.

Generated outputs or scripts:
- Use the response-systematic summary plots from Correction 2 as the current visual summary of experimental response uncertainties.

## Correction 4: Jet-To-Tau Fake Uncertainty Section

Lines:
- `4605-4611`

Original text anchor:
> "The uncertainty on the fake estimate is evaluated from variations of the fake-factor procedure."

> "See Appendix~\ref{app:tau_width_scaling} for more details."

Current status:
- The section is now present and mostly points in the right direction.
- The section title should use the same vocabulary as Chapters 7 and 8: "jet-to-tau fake background" or "jet-fake background", not "fake-jet background".
- The appendix reference `app:tau_width_scaling` is not defined in the current thesis source.
- The tau-width variation needs one important qualification: the current full-systematics run applies the validated tau-width variation to the 1-prong fake component only. The 3-prong component is left nominal because the validation target is not physically usable after simulated-background subtraction.
- Line `4611` uses old-style inline maths; convert it to `$...$`.

Reason:
- The current closure summary records three fake-source components:
  - fake-factor statistical uncertainty;
  - low-$E_{\mathrm{T}}^{\mathrm{miss}}$ control-region transfer uncertainty;
  - 1-prong tau-width fake-source composition uncertainty.
- These are not a single flat fake normalisation uncertainty.

Literature and analysis evidence:
- ATLAS high-mass $\tau+E_{\mathrm{T}}^{\mathrm{miss}}$ searches assign uncertainties to jet-background fake factors from control-region choices, non-jet background subtraction, and transfer to the signal region: https://arxiv.org/abs/2402.16576.
- The ATLAS Universal Fake Factor paper motivates fake-factor uncertainties from fake-source composition, tau-candidate $p_\mathrm{T}$, and charged-particle multiplicity: https://arxiv.org/abs/2502.04156.
- Current analysis evidence: `outputs/analysis_shadow_unfold/closure_summary.md`, sections `Fake-source systematic envelopes` and `1-prong tau-width fake-source systematic`.

Suggested correction:
- Replace the current section text with a more explicit three-component description.
- Add a compact figure if Chapter 9 needs a visual summary of fake uncertainties.

Suggested text:

```tex
\section{Uncertainty on the Jet-to-Tau Fake Background} \label{sec:fakes_uncert}

The uncertainty on the jet-to-tau fake estimate is evaluated from variations of the fake-factor procedure. This follows the treatment used in high-mass $\tau+\etmiss$ searches, where the jet background estimate is assigned uncertainties associated with the transfer of fake factors from the control region to the signal region~\cite{highmass_resonances}. The first component is the statistical uncertainty of the fake factors, propagated bin-by-bin from the ID and anti-ID event counts entering the fake-factor calculation.

The second component tests the dependence on the low-$\etmiss$ control-region definition. The fake factors are recalculated using alternative low-$\etmiss$ selections, and the largest upward and downward deviations from the nominal fake estimate are taken in each bin. This gives a bin-by-bin control-region transfer uncertainty, rather than a single overall normalisation uncertainty.

The third component probes differences in the composition of jets misidentified as tau candidates between the control and signal regions. A tau-candidate width variable is used as a proxy for differences in the fake-source composition. In the current analysis this variation is applied to the validated one-prong fake component, while the three-prong component is left at its nominal value because the corresponding validation target is not physically usable after simulated-background subtraction. This treatment follows the general observation that fake factors depend on the fake-source composition and on tau-candidate properties such as $p_\mathrm{T}^{\tau}$ and charged-particle multiplicity~\cite{Pleskot:2864863}.
```

Optional figure text:

```tex
\begin{figure}
    \centering
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{uncertainties/current/no_shadow_bin_MTW_4iter_combined_fake_source_uncertainty.png}
        \caption{Combined fake-factor statistical and low-$\etmiss$ transfer uncertainty for the nominal no-shadow response.}
        \label{fig:no_shadow_bin_MTW_combined_fake_source_uncertainty}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.47\textwidth}
        \includegraphics[width=\linewidth]{uncertainties/current/MTW_shadow_bin_250_MTW_4iter_combined_fake_source_uncertainty.png}
        \caption{Combined fake-factor statistical and low-$\etmiss$ transfer uncertainty for the $\mtw$-shadow response.}
        \label{fig:MTW_shadow_bin_250_MTW_combined_fake_source_uncertainty}
    \end{subfigure}
    \caption{Relative uncertainty on the unfolded $\mtw$ spectrum from the data-driven jet-to-tau fake estimate. The uncertainty shown combines the fake-factor statistical and low-$\etmiss$ control-region transfer components.}
    \label{fig:combined_fake_source_uncertainty_current}
\end{figure}
```

Figures/tables affected:
- No current Chapter 9 figure exists for fake-source uncertainties.
- If adding the optional figure, copy:
  - `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/fake_source_systematics/no_shadow_bin_MTW_4iter_combined_fake_source_uncertainty.png`
  - `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/fake_source_systematics/MTW_shadow_bin_250_MTW_4iter_combined_fake_source_uncertainty.png`

Generated outputs or scripts:
- `outputs/analysis_shadow_unfold/closure_summary.md`
- `outputs/analysis_shadow_unfold/plots/.../fake_source_systematics/`
- `outputs/analysis_shadow_unfold/plots/.../fake_width_systematic/`

## Correction 5: Add Method-Uncertainty Section

Lines:
- Suggested insertion after line `4611`, before Chapter 10.

Original text anchor:
> Chapter 9 currently ends after the jet-fake uncertainty section.

Current status:
- The introduction says method uncertainties are assessed through closure tests and unfolding variations.
- There is no dedicated method-uncertainty subsection explaining what this means for the current analysis.

Reason:
- The current output includes same-sample closure checks for the no-shadow and $m_{\mathrm{T}}^W$-shadow response configurations.
- The unfolding method uncertainty should distinguish bookkeeping closure from independent physics validation.
- The response-configuration comparison is relevant because the shadow response tests sensitivity to migration across the nominal lower $m_{\mathrm{T}}^W$ boundary.

Literature and analysis evidence:
- RooUnfold describes iterative Bayesian unfolding and covariance propagation: https://arxiv.org/abs/1105.1160.
- Current analysis evidence: `outputs/analysis_shadow_unfold/closure_summary.md`, section `Variable-specific shadow-bin unfolding closure summary`.

Suggested correction:
- Add a short method-uncertainty section.

Suggested text:

```tex
\section{Method Uncertainties}

Method uncertainties arise from choices made in the unfolding procedure. The main checks considered here are closure of the unfolding chain, the choice of iteration count, and the response configuration used to describe migration near the lower $\mtw$ boundary of the fiducial phase space.

Closure tests are performed by unfolding a reconstructed signal prediction and comparing the result with the corresponding fiducial truth distribution. These tests verify the consistency of the response construction and unfolding bookkeeping. Same-sample closure is not an independent modelling validation, since the reconstructed input and response are derived from the same simulated signal sample, but it is a necessary check that the unfolding procedure returns the expected spectrum under idealised conditions.

The nominal iteration choice is selected to give stable closure while avoiding unnecessary amplification of statistical fluctuations. The no-shadow and $\mtw$-shadow response configurations are compared as part of the unfolding validation, since the shadow response tests the sensitivity to migration across the lower $\mtw$ threshold.
```

Figures/tables affected:
- No figure required.
- Optional closure or covariance plots could be added, but Chapter 9 can also keep this as prose.

Generated outputs or scripts:
- `outputs/analysis_shadow_unfold/closure_summary.md`
- Representative covariance plots:
  - `outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/no_shadow_bin_MTW_4iter_covariance.png`
  - `outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/MTW_shadow_bin_250_MTW_4iter_covariance.png`

## Correction 6: Duplicate Labels, Broken Reference, And Inline-Math Cleanup

Lines:
- Duplicate labels: `4509-4594`
- Broken reference: `4609`
- Old-style inline math: `4611`

Original text anchor:
> `\label{fig:recon_uncert}`

> "See Appendix~\ref{app:tau_width_scaling} for more details."

> old-style inline $p_\mathrm{T}^{\tau}$ notation

Current status:
- The old reconstruction and trigger figure blocks contain multiple duplicate labels.
- `app:tau_width_scaling` is referenced but not defined in the current thesis source.
- Chapter 9 still contains one old-style inline math expression, despite the thesis style now using `$...$`.

Reason:
- Duplicate labels can corrupt cross-references or trigger LaTeX warnings.
- Undefined appendix references should not remain in the chapter.
- Suggested thesis text should use `$...$` for inline math.

Literature and analysis evidence:
- This is a LaTeX/source consistency issue, not a physics-literature issue.

Suggested correction:
- Removing the old figure blocks in Correction 3 fixes most duplicate labels.
- Remove the undefined appendix sentence in Correction 4 unless a valid appendix label is added.
- Replace the old-style inline expression with `$p_\mathrm{T}^{\tau}$`.

Specific duplicate labels to remove with the old blocks:
- `fig:MTW_sys_eff_pct_uncert_liny`: lines `4509`, `4515`
- `fig:TauPt_sys_eff_pct_uncert_liny`: lines `4522`, `4528`, `4534`, `4540`
- `fig:MTW_sys_TRIGGER`: lines `4558`, `4564`
- `fig:TauPt_sys_TRIGGER`: lines `4571`, `4577`, `4583`, `4589`
- `fig:recon_uncert`: lines `4544`, `4594`

Figures/tables affected:
- Old reconstruction and trigger efficiency figure blocks.

Generated outputs or scripts:
- None.

## Chapter 9 Figure And Table Checklist

This checklist is only an inventory; the corrections above describe what to do.

- TES figure block, lines `4453-4490`: replace stale `systematic_alt_binning` images with current response-systematic plots.
- Reconstruction-efficiency figure block, lines `4504-4545`: remove; stale `analysis_simple_2017` images and duplicate labels.
- Trigger-efficiency figure block, lines `4553-4595`: remove; stale `analysis_simple_2017` images and duplicate labels.
- Jet-fake uncertainty figure: optional new figure using current `fake_source_systematics` outputs.
- Method-uncertainty figures: optional; covariance plots exist, but prose is sufficient unless the chapter needs an additional visual.

## Immediate To-Do List

1. Replace the opening uncertainty-scope and experimental-section sentences using Correction 1.
2. Replace the TES prose and old TES figure block using Correction 2.
3. Replace the whole tau-efficiency subsection using Correction 3, removing the stale reconstruction and trigger figures.
4. Replace the jet-to-tau fake uncertainty section using Correction 4 and remove the undefined appendix reference.
5. Add the method-uncertainty section from Correction 5.
6. Copy current plots into `images/uncertainties/current/` if the optional figures are retained.
7. Recompile and check for duplicate labels, undefined references, and any remaining old-style inline math in Chapter 9.
