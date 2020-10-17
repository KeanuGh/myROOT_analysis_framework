import boost_histogram as bh


def scale_to_crosssection(hist: bh.Histogram, luminosity) -> None:
    """Scales histogram to cross-section. Currently undefined for """
    if len(hist.axes) > 1:
        raise Exception("Currently undefined behaviour for multi-dimentional histograms")
    hist /= luminosity
    hist /= hist.axes[0].widths
