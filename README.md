# Welcome to my framework!

So far it is able to:

- Generate Dataset objects from root files into either ROOT RDataframes or pandas Dataframes.
- Read in a cutfile and apply cuts in groups or independently. The file allows you to specify which cuts to apply and
  how, and specify which variables are going to be plotted.
- Supports holding multiple datasets.
- Supports reading off from multiple TTrees at once
- Generate a cutflow for the cuts applied, and print out cutflow histograms, a LaTeX table with cut summaries and a
  terminal printout of table.
- Plot root data in 1D Histograms with and without cuts applied, and plot overlaid cuts with corresponding acceptance
  ratios.
- Plot 2D histograms of multiple variables with cuts applied.
- Read in mass slices and normalise all to a specific luminosity.
- Many methods to apply across datasets
  _- Convert boost-histograms to ROOT TH1 histograms to access ROOT functions

# Quickstart

Initialise analysis with :

```
from src.analysis import Analysis
my_analysis = Analysis(data, analysis_label='truth_inclusive_and_slices')
```

where `data` is a dictionary containing information on your datasets. For example:

```
data = {
    'truth_inclusive': {
        'datapath': '../../data/mc16d_wmintaunu/*.root',
        'cutfile': '../../options/cutfile.txt',
        'TTree_name': 'truth',
        'lepton': 'tau'
    },
    'truth_slices': {
        'datapath': '../../data/mc16a_wmintaunu_SLICES/*.root',
        'cutfile': '../../options/cutfile.txt',
        'TTree_name': 'truth',
        'lepton': 'tau'
    }
}
```

The `Analysis` class will automatically build your datasets, which you can then access
like `my_analysis['truth_inclusive']`.

Datasets store their data in a pandas dataframe which you can access with `my_analysis['truth_inclusive'].df`.

Plotting methods are available to plot datasets with cuts, eg can do
`my_analysis.plot_with_cuts('truth_inclsive')`,
which will plot overlay of all cuts against inclusive sample, and print histograms to pickle files in output directory
named after what you passed as `analysis_label` in constructor. Variables mentioned in cutfile are automatically
generated and saved to root files.
Can then convert pickle files to ROOT files containing histograms by doing:

```
from utils.file_utils import convert_pkl_to_root
convert_pkl_to_root(conv_all=True)
```

`Analysis` class also contains methods to apply across multiple datasets. eg:

- `.merge_datasets('dataset1'. 'dataset2')` merges given datasets into the first one passed
- `.plot_hist(['dataset1', 'dataset2'], 'mu_pt', 'reco_weight')` plot same variable in multiple datasets overlayed.
  Optional ratio plot underneath
- Can use methods like `.create_subdataset(old, new, args)`, to create a new dataset based on subset of a dataset
  currently in analysis

# Cutfile

Example cutfile:

```
[CUTS]
# Name	cutstring	tree
# !!REMEMBER DEFAULT ANALYSISTOP OUTPUT IS IN MeV NOT GeV!!
Tight muon	mu_isTight = 1
eta	mu_eta.abs() < 2.4 and (mu_eta < 1.57 or mu_eta > 1.34)
E_{T}^{MISS}	met_met > 5
p_T	mu_pt > 65 and neutrino_pt > 65
M_T	mu_mt >	55
M_W	MC_WZ_m < 120	truth

[OUTPUTS]
# variables to process
PDFinfo_Q
```

Cutfile contains lists of tab-separated values.

For **[CUTS]** The columns are:

- **Name**: name of cut. This will be the label added to plots and cutflow for that cut
- **Cut String**: a boolean query to be evaluated by pandas.
- **Alt TTree**: if the given variable is in a different tree, imports and matches by including the 'eventNumber'
  TBranch

**[OUTPUTS]**

List of the variables that you plan to plot. The variables that are cut on will still be accessible in the dataframe
though so you can plot them if you want

# Dataset object

Dataset object contains the pandas DataFrame with the event data and methods to plot, cut on etc.
To initialise a Dataset you should use the `DatasetBuilder` class:

```
    builder = DatasetBuilder(
        name='example_dataset',
        TTree_name='truth,
        hard_cut='M_W'
    )
    dataset = builder.build(data_path='path/to/dataset.root', cutfile_path=path/to/cutfile.txt)
```

Can `.build()` a dataset using a pickled DataFrame as well. If a hard cut is given then the cut will be applied during
dataset generation and will not show up in the cutflow

## Build pipeline

A brief overview of the pipeline:

- Parse cutfile
- check variables and cuts
- If reading from pickle file:
    - check variables and cuts against cutfile and decide whether to rebuild from ROOT file
- Extract data from ROOT file:
    - Extract required variables from each tree and merge into given 'default' tree
    - calculate total sum of MC weights for each dataset ID and validate
    - drop any duplicate events
    - validate any events in reco tree(s) missing from truth tree(s)
    - rescale energy variables to GeV
- Calculate cuts if necessary
- Calculate reco and truth weights if necessary
- Generate cutflow
- Apply any 'hard' cuts

## Cutting

Apply cuts to dataset using the `.apply_cuts('cut 1')` method. Pass either a string or list of strings corresponding to
a cut named in the cutfile

## Metadata

The `Dataset` class comes with some useful properties and methods for variables access eg:

- `.variables` list of all variables in dataset
- `.cut_cols` list of all cuts in dataset
- `.is_truth` and `.is_reco` whether dataset contains truth or reco data
- `.n_truth_events`, `n_reco_events`, `.get_truth_events()`, `.get_reco_events()` - how many truth or reco events and
  gets the events themselves
- `.cross_section`, `.luminosity` of data
- `.dsid_metdata_printout()` prints metadata per dataset ID in dataset
- `.print_latex_table(filepath)` prints a .tex file containing table of cutflow
- `.save_pkl_file(filepath)` saves DataFrame as pickle file to filepath

## Plotting

Framework implements its own Histogram class based on [boost-histogram](https://github.com/scikit-hep/boost-histogram)
with bindings similar to ROOT TH1s

- `.plot_hist(var, bins, weight)` for simple histogram plot of variable within DataFrame
- `.plot_cut_overlays(var, bins, weight)` for overlay plot of single variable with each cut applied one after the other,
  and ratio beneath
- `.plot_mass_slice(var, bins, weight` for each dataset ID within dataframe plotted seprately
- `.gen_cutflow_hist()` generates cutflow histogram
- `.profile_plot(varx, vary)` plot of two variables against each other
  In each of these methods, 'var' and 'weight' are strings corresponding to column in DataFrame

## Extra variables

In `utils.var_helpers` I've definied a few variables aren't in AnalysisTop outputs, eg transverse mass or boson
rapidity. These can be used as variables in the cutfile and the framework will calculate them for you: extracting the
variables it needs to calculate.
So far the defined variables are:

- `mu_mt` & `e_mt` (actually boson transverse mass for different W/Z decays)
- `w_y`, `z_y`, `v_y` (W/Z rapidity. They all do the same thing)

## Subsetting

Can use methods `.subset()`, `.subset_dsid()` and `.subset_cut()` to create a new dataset based on subset of current
dataset

## Command-line script

in `run/` directory is provided the `build_dataset.py` script which can generate a single Dataset from the command line
and print cutflow and pickled DataFrame

```
$ python run/build_dataset.py -d data/my_root_data.root -n test_dataset -I truth -c options/cutfile_EXAMPLE.txt
```
