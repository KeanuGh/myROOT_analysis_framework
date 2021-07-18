from analysis.analysis import Analysis
from utils import file_utils


if __name__ == '__main__':

    # dataset inputs
    datasets = {
        'truth': {
            'datapath': '../data/*.root',
            'cutfile': '../options/cutfile_EXAMPLE.txt',
            'TTree_name': 'truth',
            'is_slices': False,
            'lepton': 'muon'
        },
        # 'truth': {
        #     'datapath': '../data/*.root',
        #     'cutfile': '../options/wpt_cuts.txt',
        #     'TTree_name': 'truth',
        #     'is_slices': False,
        #     'lepton': 'muon'
        # }
    }

    my_analysis = Analysis(datasets, analysis_label='wminmunu_test_eg', force_rebuild=False)

    # pipeline
    # my_analysis.plot_mass_slices(ds_name='truth_slices', xvar='MC_WZ_dilep_m_born', logx=True, to_pkl=True)
    my_analysis.plot_with_cuts(scaling='xs', to_pkl=False)
    my_analysis.make_all_cutgroup_2dplots(to_pkl=False)
    my_analysis.gen_cutflow_hist(all_plots=True)
    my_analysis.cutflow_printout()
    my_analysis.kinematics_printouts()
    my_analysis.print_cutflow_latex_table()
    file_utils.convert_pkl_to_root(conv_all=True)