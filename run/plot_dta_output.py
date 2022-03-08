from src.analysis import Analysis

if __name__ == '__main__':
    datasets = {
        'WTV': {
            'data_path': 'data/DTA_outputs/first_test/*',
            'cutfile_path': '../options/joanna_cutflow/DY_HM.txt',
            'lepton': 'tau',
            'label': r'$W\rightarrow\tau\nu$',
        },
    }
