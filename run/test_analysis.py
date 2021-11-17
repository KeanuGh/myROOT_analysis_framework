#!/users/keanu/miniconda3/envs/root_env/bin/python

from src.analysis import Analysis

if __name__ == '__main__':

    datasets = {
        'wminmunu': {
            'data_path': '../data/wminmunu_MC.root',
            'cutfile_path': '../options/cutfile_EXAMPLE.txt',
            'TTree_name': 'truth',
            'is_slices': 'False',
            'lepton': 'muon'
        }
    }

    my_analysis = Analysis(datasets, analysis_label='test_analysis',
                           force_rebuild=False, log_level=10, log_out='console')

    my_analysis.make_all_cutgroup_2dplots()
    my_analysis.plot_with_cuts()
