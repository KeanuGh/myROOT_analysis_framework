import boost_histogram as bh
import pytest

from utils import plotting_utils as pu


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
        actual_output = pu.get_sumw2(h)

        assert expected_output == actual_output, f"Expected: {expected_output}. Actual: {actual_output}"


class TestGetAxisLabels(object):
    # test dictionary
    test_label_xs = {
        'testvar': {
            'xlabel': 'testxlabel',
            'ylabel': 'testylabel',
        }
    }

    def test_label_read(self):
        expected_output = ('testxlabel', 'testylabel')
        actual_output = pu.get_axis_labels('testvar')

        assert expected_output == actual_output, f"Expected: {expected_output}. Actual: {actual_output}"

    def test_no_label(self):
        with pytest.warns(UserWarning) as warning:
            expected_output = (None, None)
            var_missing = 'test_var_missing'
            actual_output = pu.get_axis_labels(var_missing)

            assert warning[0].message.args[0] == f"Axis labels for {var_missing} not found in in label lookup " \
                                                 f"dictionary. Axis labels blank."
            assert expected_output == actual_output, f"Expected: {expected_output}. Actual: {actual_output}"
