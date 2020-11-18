from analysis.analysis import Analysis


if __name__ == '__main__':

    # dataset inputs
    datasets = {
        'reco': {
            'datapath': '../../data/mc16d_wmintaunu/*',
            'cutfile': '../../options/cutfile_reco.txt',
            'TTree_name': 'nominal',
            'is_slices': False,
            'lepton': 'muon'
        },
        'truth': {
            'datapath': '../../data/mc16d_wmintaunu/*',
            'cutfile': '../../options/cutfile_truth.txt',
            'TTree_name': 'truth',
            'is_slices': False,
            'lepton': 'muon'
        }
    }

    my_analysis = Analysis(datasets, analysis_label='wminmunu_reco_truth')

    # pipeline
    # my_analysis.plot_mass_slices(ds_name='truth_slices', xvar='MC_WZ_dilep_m_born', logx=True, to_pkl=True)
    # my_analysis.plot_with_cuts(scaling='xs', ds_name='truth_inclusive', to_pkl=True)
    # my_analysis.make_all_cutgroup_2dplots(ds_name='truth_inclusive', to_pkl=True)
    # my_analysis.gen_cutflow_hist(ds_name='truth_inclusive', all_plots=True)
    # my_analysis.cutflow_printout(ds_name='truth_inclusive')
    # my_analysis.kinematics_printouts()
    # my_analysis.print_cutflow_latex_table(ds_name='truth_inclusive')
    # file_utils.convert_pkl_to_root(conv_all=True)
