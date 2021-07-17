# Welcome to my framework!
So far it is able to:
- Read in many root files and concatenate.
- Read in a cutfile and apply cuts in groups or independently (todo: writeup how to use cutfile). In the file youcan specify which cuts to apply and how, and specify which variables are going to be plotted.
- Supports holding multiple datasets.
- Generate a cutflow for the cuts applied, and print out cutflow histograms, a LaTeX table with cut summaries and a terminal printout of table.
- Plot root data in 1D Histograms with and without cuts applied, and plot overlaid cuts with corresponding acceptance ratios.
- Plot 2D histograms of multiple variables with cuts applied.
- Read in mass slices and normalise all to a specific luminosity.
- Backs up cutfiles and latex table when changes are made, and saves the last analysis data in pickle files for faster readin next time round.
- Organises plots by cutfile name.
- Converts boost-histograms to ROOT TH1 histograms and back

### In the works:
- Logging: as soon as I get around to learning how to use the python logging library.
- Unit tests: Need to write many more tests. 
- Batch jobs: Need to look into using Dask for parallel processing to work with large ROOT files.

## Quickstart
Initialise analysis with a very silly import statement like:

```
from analysis.analysis import Analysis
my_analysis = Analysis(data, analysis_label='truth_inclusive_and_slices')
```

where `data` is a dictionary containing information on your datasets. For example:
```
data = {
    'truth_inclusive': {
        'datapath': '../../data/mc16d_wmintaunu/*.root',
        'cutfile': '../../options/cutfile.txt',
        'TTree_name': 'truth',
        'is_slices': False,
        'lepton': 'tau'
    },
    'truth_slices': {
        'datapath': '../../data/mc16a_wmintaunu_SLICES/*.root',
        'cutfile': '../../options/cutfile.txt',
        'TTree_name': 'truth',
        'is_slices': True,
        'lepton': 'tau'
    }
}
```
The `Analysis` class will automatically build your datasets, which you can then access like 

`my_analysis.truth_inclusive` or `my_analysis['truth_inclusive']`.
 
Datasets store their data in a pandas dataframe which you can access with `my_analysis.truth_inclusive.df`. 

Plotting methods are available to plot datasets with cuts, eg can do
`my_analysis.truth_inclusive.plot_with_cuts()` (or `my_analysis.plot_with_cuts(ds_name='truth_inclsive'`), 
which will plot overlay of all cuts against inclusive sample, and print histograms to pickle files in output directory named after what you passed as `analysis_label` in constructor.
Can then convert pickle files to ROOT files containing histograms by doing:
```
from utils.file_utils import convert_pkl_to_root
convert_pkl_to_root(conv_all=True)
```

## Cutfile
Example cutfile:
```
[CUTS]
# Name	Variable	</>	Cut (in GeV if applicable)	Group Symmetric
# !!REMEMBER DEFAULT ANALYSISTOP OUTPUT IS IN MeV NOT GeV!!
Muon $|#eta|$	MC_WZmu_el_eta_born	<	2.4	eta cut	true
Muon $p_{T}$	MC_WZmu_el_pt_born	>	25	pT cut	false
Neutrino $p_{T}$	MC_WZneutrino_pt_born	>	25	pT cut	false

[OUTPUTS]
# variables to process
MC_WZ_dilep_m_born
MC_WZ_dilep_m_bare
MC_WZ_dilep_m_dres
MC_WZ_pt

[OPTIONS]
# case-insensitive
sequential	true
```
Cutfile contains lists of tab-separated values. 

For [CUTS] The columns are:
- Name: name of cut. This will be the label added to plots and cutflow for that cut
- Variable: variable in root file to cut on
- Moreless: hopefully self-explanatory. < or > depending on what you want (doesn't handle >= and stuff .. yet)
- Cut: value to cut variable at (in GeV)
- Group: Cuts with the same 'group' value will be applied simultaneously
- Symmetric: whether or not the cut is symmetric. Eg in the example the cut on `MC_WZmu_el_eta_born` will actually be `|MC_WZmu_el_eta_born| < 2.4`

[OUTPUTS] are a list of the variables that you plan to plot. The variables that are cut on will still be accessible in the dataframe though so you can plot them if you want 

[OPTIONS] so far just contains 'sequential'. This set whether the cuts should be applied one after the other or separately. Remember if sequential is true that the order in which you write in [CUTS] matters

## Extra variables
In `utils.var_helpers` I've definied a few variables aren't in AnalysisTop outputs, eg transverse mass or boson rapidity. These can be used as variables in the cutfile and the framework will calculate them for you: extracting the variables it needs to calculate then deleting the unnecessary columns afterwards. 

So far the variables I've defined are:
- `mu_mt` & `e_mt` (actually boson transverse mass for different W/Z decays)
- `w_y`, `z_y`, `v_y` (W/Z rapidity. They all do the same thing)