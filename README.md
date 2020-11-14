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
- Logging: as soon as I get around to learning how to use the python logging library
- Unit tests: pytest stopped working for me and I haven't gotten round to fixing it yet. Bugs galore!

## Quickstart
Initialise analysis with something like.

```my_analysis = Analysis(data, analysis_label='truth_inclusive_and_slices')```

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