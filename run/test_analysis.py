#!/users/keanu/miniconda3/envs/root_env/bin/python

from src.analysis import Analysis

if __name__ == '__main__':

    datasets = {
        'wminmunu': {
            'data_path': '../data/wminmunu_MC.root',
            'cutfile_path': '../options/cutfile_EXAMPLE.txt',
            'TTree_name': 'truth',
            'year': '2015+2016',
            'lepton': 'muon',
        },
        'wmintaunu': {
            'data_path': '../data/mc16a_wmintaunu_SLICES/*.root',
            'cutfile_path': '../options/cutfile_reco.txt',
            'TTree_name': 'nominal_Loose',
            'year': '2015+2016',
            'lepton': 'tau',
            'force_rebuild': False,
            'validate_duplicated_events': False
        }
    }

    my_analysis = Analysis(datasets, analysis_label='test_analysis', log_level=10, log_out='both')

    my_analysis.plot_mass_slices('wmintaunu', 'mt')
    # my_analysis.make_all_cutgroup_2dplots('wminmunu')
    # my_analysis.plot_with_cuts('wminmunu')
