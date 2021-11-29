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
            'force_rebuild': False,
        },
        'wmintaunu': {
            'data_path': '../data/mc16a_wmintaunu_SLICES/*.root',
            'cutfile_path': '../options/cutfile_EXAMPLE.txt',
            'TTree_name': 'truth',
            'year': '2015+2016',
            'lepton': 'tau',
            'force_rebuild': True,
            'validate_duplicated_events': False
        }
    }

    my_analysis = Analysis(datasets, analysis_label='test_analysis', log_level=10, log_out='both')

    my_analysis.plot_mass_slices('wmintaunu', 'mt', bins=(50, 200, 5000))
    my_analysis.plot_mass_slices('wmintaunu', 'MC_WZ_dilep_m_born', bins=(50, 200, 5000))
    # my_analysis.make_all_cutgroup_2dplots('wminmunu')
    # my_analysis.plot_with_cuts('wminmunu')
