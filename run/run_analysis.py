from analysis.analysis import Analysis

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
    # my_analysis.plot_with_cuts(scaling='xs', to_pkl=False)
    # my_analysis.make_all_cutgroup_2dplots(to_pkl=False)
    # my_analysis.gen_cutflow_hist(all_plots=True)
    # my_analysis.cutflow_printout()
    # my_analysis.kinematics_printouts()
    # my_analysis.print_cutflow_latex_table()
    # file_utils.convert_pkl_to_root(conv_all=True)

    import utils.plotting_utils as pu
    pu.plot_1d_hist(x=[my_analysis.truth['MC_WZ_dilep_m_born'],
                       my_analysis.truth['MC_WZ_dilep_m_bare'],
                       my_analysis.truth['MC_WZ_dilep_m_dres']],
                    bins=(50, 1, 500), is_logbins=True, log_y=True,
                    x_label='dilepton mass [GeV]', filename='bornbaredres_overlay',
                    legend_label=['Born', 'bare', 'dressed'], to_file=True, legend=True, yerr='sumw2')
