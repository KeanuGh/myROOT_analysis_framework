# Chapter 5 Update Notes: Data And Monte Carlo Simulation

Source audited:
- Thesis source: `/mnt/B/Uni_Stuff_Queen_Mary/Documents/Thesis/thesis.tex`
- Chapter: `Data and Monte Carlo Simulation`
- Current chapter line range: `3246-3803`
- Analysis sample definition: `run/2017/samples.py`
- Scale-factor plotting script: `run/scale_factors.py`

Purpose:
This note records the Chapter 5 changes needed to make the thesis text match the current shadow-unfolding analysis. It is a thesis-writing note, not a production-analysis change.

## Executive Summary

Most of Chapter 5 remains structurally correct. The main updates are local:

1. Keep the corrected `25 ns` 2017 bunch spacing already present in the thesis source.
2. Explicitly describe the current `W->tau nu` sample split into `wtaunu_had` signal and `wtaunu_lep` background/MC-contamination.
3. Clarify that this hadronic-tau measurement unfolds visible tau truth variables, not light-lepton born/bare/dressed definitions.
4. Update the event-weight section so the luminosity/cross-section equation is described as the base MC normalisation, with DTA-provided per-event correction weights applied in addition.
5. Add a tau-object correction subsection.
6. Regenerate the scale-factor figures if they remain in the chapter.

## Correction 1: Data Sample Wording

Lines:
- `3250-3268`

Current status:
- The physics content is mostly correct.
- Line `3250` already says `25 ns`, which is the desired correction for the 2017 Run-2 dataset.
- Table `tab:data_period_grl` gives the full Run-2 GRL context, although this analysis uses only 2017.

Reason:
- Earlier text had been checked for the bunch-spacing issue. The current source no longer has the `50 ns` problem.
- The table is still useful context, but the prose should make clear that only 2017 is used in the measurement.

Suggested correction:
- Keep `44.3 fb^{-1}`, `25 ns`, and `13 TeV`.
- Make a small grammar correction: use "Data are separated..." rather than "Data is separated...".
- Optionally add one sentence after the GRL table reference saying that only the 2017 row enters the final analysis.

Suggested text:

```tex
This analysis uses the 2017 dataset of Run 2, corresponding to a total integrated luminosity of \SI{44.3}{\per\femto\barn}, a mean bunch spacing of \SI{25}{\nano\s}, and a centre-of-mass energy of \(\sqrt{s}=\SI{13}{\tera\electronvolt}\). Data are separated into data-taking periods, each corresponding to a particular phase of data-taking with different detector, trigger, and luminosity conditions. The use of a single-year dataset in this analysis is due to time and resource constraints.
```

Figures/tables affected:
- `tab:data_period_grl`, lines `3254-3268`.
- No regenerated table is required.

Generated outputs or scripts:
- None.

## Correction 2: Signal Sample And Tau-Decay Categorisation

Lines:
- `3284-3286`
- `3288-3329`, figure `fig:signal`

Current status:
- The Sherpa 2.2.11 signal-generation description is still broadly correct.
- The text does not yet make the current hadronic/leptonic tau split explicit.

Reason:
- The current analysis separates the inclusive `W->tau nu` MC into:
  - `wtaunu_had`: `TruthTau_isHadronic`, treated as the hadronic-tau signal;
  - `wtaunu_lep`: `!(TruthTau_isHadronic)`, treated as background/MC-contamination.
- This split should be visible in Chapter 5 because it defines which simulated samples are signal and which are background.

Suggested correction:
- Add a paragraph after the high-mass sample paragraph.
- Use `TruthBosonM` when describing the low/high sample split, because this is the quantity used in `run/2017/samples.py`.

Suggested text:

```tex
In the current analysis workflow the inclusive \(W\rightarrow\tau\nu\) simulated sample is separated according to the truth tau decay mode. Events with \lstinline{TruthTau_isHadronic} form the \(W\rightarrow\tau\nu\rightarrow\mathrm{hadrons}\) signal sample, labelled \lstinline{wtaunu_had}. Events in which the tau decays leptonically are assigned to a separate \lstinline{wtaunu_lep} sample and are treated as a simulated background contribution rather than part of the fiducial hadronic-tau signal. This separation is applied consistently to both the low-mass \(\lstinline{TruthBosonM}<\SI{120}{\GeV}\) and high-mass Sherpa samples.
```

Figures/tables affected:
- `fig:signal`, lines `3288-3329`: no redraw needed.
- Caption line `3327`: soften the precise `99.99%` statement unless a citation is added.

Suggested caption adjustment:

```tex
Examples of hadronically decaying taus in charged Drell-Yan processes. Hadronically decaying taus are dominated by final states with either one or three charged particles. In these Feynman diagrams, the crossed circle represents the effective operator for the hadronisation process \(q\bar{q}\rightarrow\) mesons.
```

Generated outputs or scripts:
- Evidence in `run/2017/samples.py:43-67`.
- No figure generation needed.

## Correction 3: Light-Leptonic Charged Drell-Yan Background

Lines:
- `3333-3348`

Current status:
- This subsection is useful and should stay.
- The current wording has a few problems:
  - "separate at (`mtw < 120 GeV`)" is unclear;
  - it should refer to `TruthBosonM < 120 GeV`;
  - it should explicitly connect to `wtaunu_lep`, `wlnu`, and direct light-lepton charged DY backgrounds.

Reason:
- The current sample configuration has:
  - `wtaunu_lep`, lines `56-67` in `run/2017/samples.py`;
  - `wlnu`, lines `68-81`;
  - both low-mass and high-mass components.

Suggested correction:
- Replace the prose at lines `3335-3337`.
- Keep the standalone light-leptonic charged DY diagram only if the duplicated LaTeX label is fixed.

Suggested text:

```tex
The first background is the light-leptonic counterpart to the charged Drell-Yan signal. This includes direct \(W\rightarrow e\nu\) and \(W\rightarrow\mu\nu\) production, as well as \(W\rightarrow\tau\nu\) events in which the tau decays leptonically. The same Sherpa 2.2.11 generation strategy is used, with low-mass samples selected using \lstinline{TruthBosonM < 120 GeV} and merged with dedicated high-mass samples. These events can enter the selected region when an electron, muon, or leptonic-tau decay product is reconstructed in a way that passes the hadronic-tau event selection.
```

Figures/tables affected:
- Standalone figure lines `3339-3348`.
- The label `subfig:light_lepton_background` is duplicated inside `fig:v_plus_jets_bkg` at lines `3356-3429`.

Action:
- Rename one of the duplicate labels, for example:
  - standalone figure: `fig:light_leptonic_charged_dy`;
  - subfigure inside V+jets: keep `subfig:light_lepton_background`.

Generated outputs or scripts:
- Evidence in `run/2017/samples.py:56-80`.
- No generated plot needed.

## Correction 4: V(+Jets) And Neutral DY Description

Lines:
- `3350-3429`
- Specific caption issue at `3375`

Current status:
- The generator description is still correct.
- The sample grouping should be made a little more explicit.
- The neutral DY caption currently implies that a tau from `Z->tau tau` is "misidentified"; this is too loose.

Reason:
- In the current analysis, `wlnu` and `zll` are grouped samples in `run/2017/samples.py`.
- `Z->tau tau` can be a real-tau background, while light leptons and jets can be misidentified as hadronic tau candidates.

Suggested correction:
- Add one sentence after line `3354` explaining the current grouped sample categories.
- Replace the caption sentence around line `3375`.

Suggested text:

```tex
In the current analysis configuration, the charged light-lepton samples are grouped as \lstinline{wlnu}, while neutral Drell-Yan samples, including \(Z\rightarrow ee\), \(Z\rightarrow\mu\mu\), \(Z\rightarrow\tau\tau\), and \(Z\rightarrow\nu\nu\), are grouped as \lstinline{zll}.
```

Suggested caption replacement for line `3375`:

```tex
Neutral Drell-Yan background. The \(Z/\gamma^*\) can decay into charged leptons or neutrinos. \(Z\rightarrow\tau\tau\) can enter through a reconstructed hadronic tau, while light leptons or jets may also be misidentified as tau candidates.
```

Figures/tables affected:
- `fig:v_plus_jets_bkg`, lines `3356-3429`: no redraw needed.

Generated outputs or scripts:
- Evidence in `run/2017/samples.py:68-98`.

## Correction 5: Top And Diboson Backgrounds

Lines:
- Top: `3434-3436`, figure `fig:top_bkg` at `3438-3521`
- Diboson: `3525`, figure `fig:diboson_bkg` at `3527-3600`

Current status:
- These sections are broadly consistent with the current sample configuration.
- No urgent rewrite is required.

Reason:
- Current top sample paths include `PP8_singletop`, `PP8_tchan`, `PP8_Wt_DR_dilepton`, and `PP8_ttbar_hdamp258p75`.
- Current diboson sample paths include Sherpa 2.2.12 fully leptonic samples and Sherpa 2.2.11 semileptonic samples.

Suggested correction:
- Optional precision edit only. If desired, add one sentence to each section summarising the current grouped sample names.

Suggested top text:

```tex
The current analysis groups the simulated top backgrounds into a single \lstinline{top} sample, containing \(t\bar{t}\), single-top, \(t\)-channel, and \(Wt\) production.
```

Suggested diboson text:

```tex
The current analysis groups the simulated diboson backgrounds into a single \lstinline{diboson} sample, containing fully leptonic and semileptonic diboson final states.
```

Figures/tables affected:
- `fig:top_bkg`: no redraw needed.
- `fig:diboson_bkg`: no redraw needed.

Generated outputs or scripts:
- Evidence in `run/2017/samples.py:99-123`.

## Correction 6: Truth Definition For Hadronic-Tau Unfolding

Lines:
- `3602-3715`
- Figures `fig:bornbaredres`, `fig:dilepoverlay`, `fig:bornbareratio`

Current status:
- The born/bare/dressed section is useful general Drell-Yan truth-definition context.
- It is light-lepton oriented.
- The current hadronic-tau measurement does not unfold a light-lepton born, bare, or dressed definition directly.

Reason:
- The current unfolding response uses visible hadronic-tau truth variables, including `VisTruthTauPt` and `TruthMTW`.
- Without clarification, the reader could wrongly infer that the hadronic tau fiducial definition is a born/bare/dressed light-lepton definition.

Suggested correction:
- Keep the section, but add a paragraph after line `3715`.

Suggested text:

```tex
For the hadronic-tau measurement in this thesis, the unfolded fiducial observables are not defined using the light-lepton born, bare, or dressed four-vector directly. Instead, the tau kinematics are built from the visible hadronic tau decay products, while the transverse mass uses the corresponding truth-level missing transverse momentum from neutrinos in the event. In the analysis code this is represented by variables such as \lstinline{VisTruthTauPt} and \lstinline{TruthMTW}. The born, bare, and dressed definitions remain useful context for how truth-level lepton definitions are treated in Drell-Yan measurements, but the hadronic-tau fiducial definition is based on visible tau decay products.
```

Figures/tables affected:
- `fig:bornbaredres`, lines `3603-3681`: no redraw needed.
- `fig:dilepoverlay`, lines `3682-3687`: no current generated replacement identified.
- `fig:bornbareratio`, lines `3688-3700`: no current generated replacement identified.

Generated outputs or scripts:
- Current hadronic-tau unfolding outputs are under `outputs/analysis_shadow_unfold/`.
- No direct regeneration script is identified for the born/bare/dressed conceptual figures.

## Correction 7: MC Correction And Event-Weight Wording

Lines:
- Introduction: `3742-3744`
- Equation and explanation: `3781-3787`

Current status:
- The broad correction categories are still correct.
- The event-weight equation is too compressed for the current workflow.
- It reads as if the full final event weight is only the generator weight and metadata normalisation.

Reason:
- The current framework applies luminosity, cross-section, filter-efficiency, k-factor, and sum-of-weights metadata together with DTA-provided per-event weights.
- The DTA-derived weight includes event-level corrections that should be described carefully without overclaiming that every component is recalculated in the thesis analysis code.

Suggested correction:
- Recast Eq. `eq:event_weight` as the base MC normalisation.
- Add text that the final event weight multiplies this base normalisation by generator and per-event correction factors.

Suggested text:

```tex
The base MC normalisation rescales simulated events to the integrated luminosity of the analysed data period:
\begin{equation} \label{eq:event_weight}
    w_{\mathrm{base}} =
    \frac{\mathcal{L}_{\mathrm{data}}\sigma_{\mathrm{MC}}\epsilon_F k}{\sum w_{\mathrm{MC}}}.
\end{equation}
Here \(\mathcal{L}_{\mathrm{data}}\) is the integrated luminosity, \(\sigma_{\mathrm{MC}}\) is the generated process cross-section, \(\epsilon_F\) is the generator filter efficiency, \(k\) is the higher-order correction factor, and \(\sum w_{\mathrm{MC}}\) is the sum of generator weights for the sample. In the final event weight this normalisation is multiplied by the generator event weight and the relevant per-event correction factors stored in the derived ntuples, including pileup, trigger, tau-object, and jet-vertex-tagging scale factors where applicable.
```

Figures/tables affected:
- None directly.

Generated outputs or scripts:
- Evidence in `src/datasetbuilder.py`.
- Current 2017 luminosity: `44307.4 pb^{-1}`.

## Correction 8: Generator-Weight And Pileup-Reweighting Figures

Lines:
- Figure block: `3751-3766`
- `fig:mc_weighted_log`
- `fig:prw_weighted`

Current status:
- These figures are still relevant if Chapter 5 keeps a visual explanation of generator weights and pileup reweighting.
- Current repo copies are old:
  - `outputs/scale_factors/plots/mc_weighted.png`: `2024-12-01`
  - `outputs/scale_factors/plots/mc_weighted_log.png`: `2024-12-01`
  - `outputs/scale_factors/plots/prw_weighted.png`: `2024-12-01`

Reason:
- The thesis should not use stale plots if the analysis framework and sample bookkeeping have changed.

Suggested correction:
- Regenerate the figures if they remain in the chapter.
- If the regenerated plots are only illustrative and not central to the corrected result, the captions should say they demonstrate the effect of individual event-weight components, not the complete event-weight model.

Suggested caption adjustment:

```tex
Comparison between unweighted reconstructed \(\mtw\) and the same distribution after applying the generator-weight component for a 2017 \(W\rightarrow\tau\nu\) simulated sample.
```

For PRW:

```tex
Comparison between unweighted reconstructed \(\mtw\) and the same distribution after applying the pileup-reweighting component for a 2017 \(W\rightarrow\tau\nu\) simulated sample.
```

Figures/tables affected:
- `fig:mc_weighted_log`, lines `3751-3758`.
- `fig:prw_weighted`, lines `3759-3766`.

Generated outputs or scripts:
- Script:
  ```bash
  pixi run python run/scale_factors.py
  ```
- Expected outputs:
  ```text
  outputs/scale_factors/plots/mc_weighted.png
  outputs/scale_factors/plots/mc_weighted_log.png
  outputs/scale_factors/plots/prw_weighted.png
  ```
- The script reads DTA input files and writes to `outputs/scale_factors/`.

## Correction 9: Trigger, Tau-Object, And Jet-Fake Separation

Lines:
- Trigger scale factors: `3795-3797`
- Insert new tau-object subsection before `3799`

Current status:
- The trigger scale-factor paragraph is relevant but incomplete.
- Chapter 5 does not currently describe the tau-object correction chain applied to simulated genuine hadronic tau candidates.
- The existing wording should be tightened so it does not imply that jet-to-tau misidentification is corrected with the same true-tau object scale factors.

Reason:
- ATLAS tau performance work treats hadronic tau reconstruction, identification, trigger efficiency, and energy calibration as dedicated tau-object performance ingredients measured or constrained with data.
- The high-mass \(W'\rightarrow\tau\nu\) ATLAS analysis reconstructs \(\tau_{\mathrm{had-vis}}\) candidates from visible hadronic tau decay products, applies tau identification with an RNN discriminant, and separates jet-background estimation from non-jet simulated backgrounds.
- The ATLAS Universal Fake Factor paper explicitly motivates data-driven methods for jets misidentified as tau leptons, because those backgrounds are not reliably modelled by simulation.
- In the current analysis code, tau energy scale and tau-efficiency variations are propagated through the response, while raw `TAUS_TRUEHADTAU_EFF_RNNID` and `TAUS_TRUEHADTAU_EFF_JETID` variations are excluded from the response-systematic set because jet-to-tau misidentification is modelled by the data-driven fake-factor estimate.

Literature check:
- ATLAS tau performance at \(\sqrt{s}=8\) TeV documents the trigger, offline reconstruction, identification, and energy calibration algorithms for hadronic tau decays, including measured tau ID, tau trigger, and tau energy scale performance: https://arxiv.org/abs/1412.7086.
- The Run-2 ATLAS charged-pion response measurement uses charged pions from tau decays to constrain calorimeter response in data and simulation, supporting the need to treat tau/charged-hadron energy response as a calibrated object uncertainty: https://arxiv.org/abs/2108.09043.
- The high-mass \(\tau+E_{\mathrm{T}}^{\mathrm{miss}}\) resonance search describes \(\tau_{\mathrm{had-vis}}\) reconstruction from visible tau decay products, RNN tau identification, and a separate data-driven jet-background estimate: https://arxiv.org/abs/2402.16576.
- The ATLAS Universal Fake Factor paper states that jets misidentified as tau leptons are not reliably modelled by MC and are therefore estimated with data-driven fake factors: https://arxiv.org/abs/2502.04156.

Suggested correction:
- Keep the trigger paragraph, but connect it to the broader tau-object correction chain.
- Add a new subsubsection before JVT/fJVT.
- Use "simulated genuine hadronic tau candidates" or "simulated true hadronic tau candidates" rather than plain "tau candidates". This avoids mixing true-tau efficiency corrections with jet-to-tau fake modelling.
- Mention tau energy calibration/TES separately from efficiency scale factors.
- Mention electron-overlap-removal only as one of the available tau-efficiency correction components in this analysis, not as a universal standalone ATLAS tau correction.
- State that jet-to-tau misidentification is not corrected with ordinary true-tau scale factors; it is handled by the data-driven fake-factor method.

Suggested text:

```tex
\subsubsection{Tau Reconstruction, Identification, Trigger, and Energy Calibration Corrections}

Simulated genuine hadronic tau candidates require object-level corrections to account for residual differences between data and simulation. These include corrections for tau reconstruction and identification efficiencies, tau-trigger efficiencies, and available overlap-removal efficiency components such as electron-overlap removal. The tau energy calibration is treated separately through tau energy scale variations. The corresponding tau-efficiency and tau-energy-scale variations are propagated as systematic uncertainties in the reconstructed-to-fiducial response.

These corrections apply to simulated genuine hadronic tau candidates. They are distinct from the background in which a quark- or gluon-initiated jet is reconstructed as a tau candidate. Such jet-to-tau misidentification is not modelled with ordinary true-tau efficiency scale factors in this analysis; it is estimated with the data-driven fake-factor method.
```

Figures/tables affected:
- No Chapter 5 figure is required.

Generated outputs or scripts:
- Code evidence:
  ```text
  src/dataset.py
  src/datasetbuilder.py
  run/2017/analysis_shadow_unfold.py
  run/2017/shadow_unfold/systematics.py
  ```
- Latest full-systematics log:
  ```text
  outputs/analysis_shadow_unfold/logs/analysis_shadow_unfold_2026-06-24_18-14-32.log
  ```
- Response-systematics plots, useful as supporting evidence but not necessarily Chapter 5 figures:
  ```text
  outputs/analysis_shadow_unfold/plots/no_shadow_bin/MTW/systematics/no_shadow_bin_MTW_response_systematics.png
  outputs/analysis_shadow_unfold/plots/MTW_shadow_bin_250/MTW/systematics/MTW_shadow_bin_250_MTW_response_systematics.png
  ```

## Chapter 5 Figure And Table Checklist

This checklist is only an inventory; the corrections above describe what to do.

- `tab:data_period_grl`, lines `3254-3268`: keep, optional prose clarification.
- `fig:signal`, lines `3288-3329`: keep, soften caption if uncited.
- Standalone light-leptonic DY figure, lines `3339-3348`: keep only after fixing duplicate label.
- `fig:v_plus_jets_bkg`, lines `3356-3429`: keep, adjust neutral DY caption wording.
- `fig:top_bkg`, lines `3438-3521`: keep.
- `fig:diboson_bkg`, lines `3527-3600`: keep.
- `fig:bornbaredres`, lines `3603-3681`: keep as context.
- `fig:dilepoverlay`, lines `3682-3687`: keep as context; no current regenerated replacement.
- `fig:bornbareratio`, lines `3688-3700`: keep as context; no current regenerated replacement.
- `fig:mc_weighted_log`, lines `3751-3758`: regenerate if retained.
- `fig:prw_weighted`, lines `3759-3766`: regenerate if retained.

## Immediate To-Do List

1. Apply Corrections 1-7 and 9 as thesis text edits.
2. Fix the duplicate LaTeX label `subfig:light_lepton_background`.
3. Regenerate scale-factor figures if retaining them:
   ```bash
   pixi run python run/scale_factors.py
   ```
4. Recompile the thesis and inspect Chapter 5 figure placement, especially the correction-plot block at lines `3751-3766`.
