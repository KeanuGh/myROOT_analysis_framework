import uproot4 as uproot
import matplotlib.pyplot as plt
import boost_histogram as bh
import mplhep as hep
import time
import numpy as np
import numba


@numba.jit()
def divide_by_binwidth(hist):
    hist = hist / hist.axes.widths
    return hist


def main():
    # mplhep Settings
    plt.style.use([hep.style.ATLAS,
                   {'font.sans-serif': ['Tex Gyre Heros']},  # use when helvetica isn't installed
                   {'axes.labelsize': 23},
                   {'axes.labelpad': 23},
                   {'axes.titlelocation': 'right'},
                   ])

    filename = "../data/wminmunu_MC.root"

    # extract data
    print(f"opening file {filename}...")
    start = time.time()
    truth = uproot.open(filename)["truth"]
    end1 = time.time()
    print(f"took {end1 - start:.3g}s to extract truth.")

    dilep_pt_born = truth["MC_WZ_dilep_pt_born"].array() / 1000  # GeV

    # cross-section and luminosity calculation
    print("calculating luminosity...")
    weight_mc = truth["weight_mc"]
    print(f"weight_mc: {weight_mc}")
    # weights = np.fromiter((1 if i > 0 else -1 for i in weight_mc), float)  # this line is bottleneck
    weights = weight_mc.array().apply(lambda x: -1 if x < 0 else 1, convert_dtype=True)
    print(f"weights: {weights}")
    cross_section = sum(np.absolute(weight_mc)) / len(weight_mc)
    print(f"cross-section: {cross_section:.2f} fb")
    lumi = weights.sum() / cross_section
    print(f"luminosity: {lumi:.2f} fb-1, {type(lumi)}")

    # setup log histogram
    print("setting up histogram...")
    hdilep_pt_born = bh.Histogram(bh.axis.Regular(30, 1, 500, transform=bh.axis.transform.log))

    # fill histogram
    print("filling histogram...")
    hdilep_pt_born.fill(dilep_pt_born, weight=weights)
    print(f"number of events: {hdilep_pt_born.sum()}")
    print(f"including flow: {hdilep_pt_born.sum(flow=True)}")

    hdilep_pt_born /= float(lumi)
    hdilep_pt_born /= hdilep_pt_born.axes[0].widths

    hep.histplot(hdilep_pt_born)
    hep.atlas.label(data=False, paper=False, year='2020')
    plt.semilogy()
    plt.xlabel(r"Born $m_{ll}$ [GeV]")
    plt.ylabel(r"$\frac{d\sigma}{dm_{ll}}$ [fb GeV$^{-1}$]")
    plt.show()


if __name__ == '__main__':
    main()
