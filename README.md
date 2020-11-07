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

### In the works:
- Logging: as soon as I get around to learning how to use the python logging library
- Unit tests: pytest stopped working for me and I haven't gotten round to fixing it yet. Bugs galore!