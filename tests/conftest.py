import numpy as np
import pytest
import uproot


@pytest.fixture()
def tmp_root_datafile(tmpdir):
    """Generate test root file to read in"""
    datapath = tmpdir.join('test_data.root')
    with uproot.recreate(datapath) as test_file:
        test_file['tree1'] = {'testvar1': np.arange(1000),
                              'testvar2': np.arange(1000) * 1.1,
                              'testvar3': np.arange(1000) * 3,
                              'weight_mc': np.append(np.ones(990), -1*np.ones(10)),
                              'mcChannelNumber': np.append(np.ones(500), np.full(500, 2)),
                              }
        test_file['sumWeights'] = {'totalEventsWeighted': np.array([980]),
                                   'dsid': np.array([1])}
    yield str(datapath)


@pytest.fixture()
def tmp_root_datafiles(tmpdir):
    """Generate 3 test root file to read in"""
    for i in [1, 2, 3]:
        datapath = tmpdir.join(f'test_data{i}.root')
        with uproot.recreate(datapath) as test_file:
            test_file['tree1'] = {'testvar1': np.arange(1000 * i),
                                  'testvar2': np.arange(1000 * i) * 1.1 * i,
                                  'testvar3': np.arange(1000 * i) * i,
                                  'weight_mc': np.append(np.ones(990 * i), -1 * np.ones(10 * i)),
                                  'mcChannelNumber': np.full(1000 * i, i),
                                  }
            test_file['sumWeights'] = {'totalEventsWeighted': np.array([980 * i]),
                                       'dsid': [i]}
    yield str(tmpdir.join(f'test_data*.root'))

