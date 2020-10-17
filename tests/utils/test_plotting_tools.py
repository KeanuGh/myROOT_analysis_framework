import boost_histogram as bh
from utils.plotting_tools import get_sumw2_1d


# class TestScaleToCrosssection(object):
#     def test_boosthistogram_scale:
#

class TestGetSumW21D(object):
    def test_weight_output(self):
        # Make 1-d histogram with 5 logarithmic bins from 1e0 to 1e5
        h = bh.Histogram(
            bh.axis.Regular(5, 1e0, 1e5, metadata="x", transform=bh.axis.transform.log),
            storage=bh.storage.Weight(),
        )

        # Fill histogram with numbers
        x = (2e0, 3e1, 2e2, 2e3, 2e4)

        # Doing this several times so the variance is more interesting
        h.fill(x, weight=2)
        h.fill(x, weight=2)
        h.fill(x, weight=2)
        h.fill(x, weight=2)

        expected_output = [4.0, 4.0, 4.0, 4.0, 4.0]
        actual_output = get_sumw2_1d(h)

        assert expected_output == actual_output, f"Expected: {expected_output}. Actual: {actual_output}"
