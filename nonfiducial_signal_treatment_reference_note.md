# Nonfiducial Signal Treatment Reference Note

## Question

We are considering subtracting reconstructed `wtaunu_had` events that pass the reconstructed selection but fail the nominal truth-fiducial definition before unfolding. The purpose is to make the measured detector-level input correspond to the fiducial truth-level quantity being unfolded.

This note identifies an ATLAS reference that supports the analysis principle and describes how their implementation relates to ours.

## Recommended Reference

**ATLAS Collaboration**, *Integrated and differential fiducial cross-section measurements for the vector boson fusion production of the Higgs boson in the \(H \rightarrow WW^{\ast}\rightarrow e\nu\mu\nu\) decay channel at 13 TeV with the ATLAS detector*, Phys. Rev. D 108, 072003, arXiv:2304.03053.

Reference list entry:

```bibtex
@article{ATLAS:2023zrv,
  author        = "{ATLAS Collaboration}",
  title         = "{Integrated and differential fiducial cross-section measurements for the vector boson fusion production of the Higgs boson in the H -> WW* -> e nu mu nu decay channel at 13 TeV with the ATLAS detector}",
  journal       = "Phys. Rev. D",
  volume        = "108",
  number        = "7",
  pages         = "072003",
  year          = "2023",
  doi           = "10.1103/PhysRevD.108.072003",
  eprint        = "2304.03053",
  archivePrefix = "arXiv",
  primaryClass  = "hep-ex"
}
```

## What ATLAS Does

This ATLAS analysis measures VBF \(H\rightarrow WW^{\ast}\rightarrow e\nu\mu\nu\) cross-sections in a fiducial phase space. The paper states that:

- the measured signal cross-section is corrected/unfolded for detector inefficiency and resolution;
- the fiducial region is chosen close to the experimental event selection to minimise model dependence;
- the unfolding to particle level is implemented using detector response corrections directly in the likelihood fit.

So ATLAS does not simply unfold every reconstructed signal-like event as if it belonged to the fiducial truth target. Their fit contains a detector-response model connecting detector-level yields to the particle-level fiducial cross-section being measured.

## How This Maps to Our Case

Our measurement target is a fiducial \(W\rightarrow\tau\nu\rightarrow\mathrm{hadrons}\) cross-section. The response matrix maps nominal truth-fiducial hadronic-tau events to reconstructed selected events.

When the reconstructed selection is enlarged with shadow bins, the detector-level `wtaunu_had` sample contains:

```text
fiducial signal reconstructed in the selected region
+ signal generated outside the nominal truth fiducial region
+ other backgrounds
+ fake tau contribution
```

The nonfiducial signal component is not part of the fiducial truth cross-section being measured. If the response matrix has no explicit out-of-fiducial truth/fake category, this component should be treated as a background-like correction to the measured input:

```text
input to unfolding =
  data
  - ordinary backgrounds
  - fake-tau estimate
  - reconstructed nonfiducial signal
```

For signal-MC closure, the analogous test is:

```text
closure input =
  reconstructed signal
  - reconstructed nonfiducial signal
```

where:

```text
reconstructed nonfiducial signal =
  wtaunu_had MC passing the reconstructed selection
  and failing the nominal truth fiducial selection
```

## Why This Is Defensible

The key point is the fiducial definition. The result is not intended to measure all generator-level \(W\rightarrow\tau\nu\rightarrow\mathrm{hadrons}\) events that happen to reconstruct into the analysis selection. It is intended to measure the nominal truth-fiducial phase space.

Therefore, reconstructed signal events outside that truth-fiducial phase space are nonfiducial contamination for this measurement. Treating them before unfolding is the same principle as the ATLAS likelihood-response approach: detector-level selected events are related to the particle-level fiducial target through a response/correction model, rather than assumed to be one homogeneous fiducial signal population.

## Important Distinction

ATLAS implements the correction in a likelihood fit with detector-response corrections. Our current framework uses a histogram subtraction followed by RooUnfold. These are not identical implementations.

The common analysis principle is:

```text
Only the fiducial signal component should be unfolded to the fiducial truth result.
Detector-level signal-like events outside that fiducial target must be either:
  1. modelled explicitly in the response/fit, or
  2. subtracted as nonfiducial signal contamination before unfolding.
```

Given our current RooUnfold response does not include an explicit out-of-fiducial category, option 2 is the cleaner and more transparent implementation.

## Additional Procedural Precedent

An older ATLAS \(H\rightarrow WW^{\ast}\rightarrow e\nu\mu\nu\) fiducial differential cross-section paper, arXiv:1604.02997, used a background-subtraction plus iterative Bayesian unfolding workflow. It states that VBF and \(VH\) Higgs contributions were subtracted assuming the Standard Model expectation when measuring the gluon-fusion fiducial cross-section, and that dominant backgrounds were estimated, subtracted from data, and then detector effects were corrected with an iterative Bayesian method.

That is not the same as our nonfiducial subtraction, but it is useful precedent for treating signal-like processes outside the chosen measurement target as subtracted components before unfolding.

