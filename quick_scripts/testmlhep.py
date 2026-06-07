import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

dy = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0.354797,
        0.177398,
        2.60481,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.177398,
        0.177398,
        0,
        0.177398,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.177398,
        0,
        0,
        0,
        0,
    ]
)
ttbar = np.array(
    [
        0.00465086,
        0,
        0.00465086,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.00465086,
        0,
        0,
        0,
        0,
        0,
        0.00465086,
        0,
        0,
        0,
        0,
        0.00465086,
        0.00465086,
        0,
        0,
        0.0139526,
        0,
        0,
        0.00465086,
        0,
        0,
        0,
        0.00465086,
        0.00465086,
        0.0139526,
        0,
        0,
    ]
)
zz = np.array(
    [
        0.181215,
        0.257161,
        0.44846,
        0.830071,
        1.80272,
        4.57354,
        13.9677,
        14.0178,
        4.10974,
        1.58934,
        0.989974,
        0.839775,
        0.887188,
        0.967021,
        1.07882,
        1.27942,
        1.36681,
        1.4333,
        1.45141,
        1.41572,
        1.51464,
        1.45026,
        1.47328,
        1.42899,
        1.38757,
        1.33561,
        1.3075,
        1.29831,
        1.31402,
        1.30672,
        1.36442,
        1.39256,
        1.43472,
        1.58321,
        1.85313,
        2.19304,
        2.95083,
    ]
)
hzz = np.array(
    [
        0.00340992,
        0.00450225,
        0.00808944,
        0.0080008,
        0.00801578,
        0.0108945,
        0.00794274,
        0.00950757,
        0.0130648,
        0.0163568,
        0.0233832,
        0.0334813,
        0.0427229,
        0.0738129,
        0.13282,
        0.256384,
        0.648352,
        2.38742,
        4.87193,
        0.944299,
        0.155005,
        0.0374193,
        0.0138906,
        0.00630364,
        0.00419265,
        0.00358719,
        0.00122527,
        0.000885718,
        0.000590479,
        0.000885718,
        0.000797085,
        8.86337e-05,
        0.000501845,
        8.86337e-05,
        0.000546162,
        4.43168e-05,
        8.86337e-05,
    ]
)
# Data for later use.
file_names = [
    "4e_2011.csv",
    "4mu_2011.csv",
    "2e2mu_2011.csv",
    "4mu_2012.csv",
    "4e_2012.csv",
    "2e2mu_2012.csv",
]
basepath = "https://raw.githubusercontent.com/GuillermoFidalgo/Python-for-STEM-Teachers-Workshop/master/data/"

# here we have merged them into one big list and simultaneously convert it into a pandas dataframe.
csvs = [pd.read_csv(f"{basepath}{file_name}") for file_name in file_names]

fourlep = pd.concat(csvs)

rmin = 70
rmax = 181
nbins = 37

M_hist = np.histogram(fourlep["M"], bins=nbins, range=(rmin, rmax))
# the tuple `M_hist` that this function gives is so common in python that it is recognized by mplhep plotting functions


hist, bins = M_hist  # hist=frequency ; bins=Mass values
width = bins[1] - bins[0]

center = (bins[:-1] + bins[1:]) / 2

label = [r"$t\bar{t}$", "Z/$\gamma^{*}$ + X", r"ZZ $\rightarrow$ 4l"]
xerrs = [width * 0.5 for i in range(0, nbins)]
yerrs = np.sqrt(hist)
fig, ax = plt.subplots(figsize=(10, 5))

hep.histplot(
    [ttbar, dy, zz, hzz],
    stack=True,
    bins=bins,
    histtype="fill",
    color=["grey", "g", "b", "w"],
    alpha=[0.5, 0.5, 0.5, 1],
    edgecolor=["orange", "b", "g", "r"],
    linewidth=1,
    # linestyle="-",
    label=[
        r"$t\bar{t}$",
        "Z/$\gamma^{*}$ + X",
        r"ZZ $\rightarrow$ 4l",
        "$m_{H}$ = 125 GeV",
    ],
    ax=ax,
)

hep.cms.label(rlabel="")

# Measured data
ax.errorbar(
    center, hist, xerr=xerrs, yerr=yerrs, linestyle="None", color="black", marker="o", label="Data"
)

ax.set_title(
    "$ \sqrt{s} = 7$ TeV, L = 2.3 $fb^{-1}$; $\sqrt{s} = 8$ TeV, L = 11.6 $fb^{-1}$ \n",
    fontsize=16,
)
ax.set_xlabel("$m_{4l}$ (GeV)", fontsize=15)
ax.set_ylabel("Events / 3 GeV\n", fontsize=15)
ax.set_ylim(0, 25)
ax.set_xlim(rmin, rmax)
ax.legend(fontsize=15)

fig.savefig("final-plot.png", dpi=140)
fig.show()
