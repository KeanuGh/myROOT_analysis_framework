import boost_histogram as bh
import mplhep as hep

# Make 1-d histogram with 5 logarithmic bins from 1e0 to 1e5
h = bh.Histogram(
    bh.axis.Regular(5, 1e0, 1e5, metadata="x", transform=bh.axis.transform.log),
    storage=bh.storage.Weight(),
)

# Fill histogram with numbers
x = (2e0, 2e1, 2e2, 2e3, 2e4)
h.fill(x, weight=2)

print(h.view().value)
print(type(h.view().value))
hep.histplot(h.view().value, bins=h.axes[0].edges)
