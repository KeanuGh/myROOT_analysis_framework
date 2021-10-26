from src.analysis import Analysis

if __name__ == '__main__':

    # dataset inputs
    datasets = {
        'wmintaunu_nominal': {
            'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wmintaunu*/*.root',
            'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
            'TTree_name': 'nominal_Loose',
            'is_slices': True,
            'lepton': 'tau'
        },
        'wplustaunu_nominal': {
            'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wplustaunu*/*.root',
            'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
            'TTree_name': 'nominal_Loose',
            'is_slices': True,
            'lepton': 'tau'
        },
        'wminmunu_nominal': {
            'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wminmunu*/*.root',
            'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
            'TTree_name': 'nominal_Loose',
            'is_slices': True,
            'lepton': 'muon'
        },
        'wplusmunu_nominal': {
            'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wplusmunu*/*.root',
            'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
            'TTree_name': 'nominal_Loose',
            'is_slices': True,
            'lepton': 'muon'
        },
    }

    my_analysis = Analysis(datasets, analysis_label='jesal_cutflow', force_rebuild=True, log_level=10)

    # pipeline
    my_analysis.gen_cutflow_hist(event=True)
    my_analysis.plot_mass_slices(xvar='mu_mt', logx=True, to_pkl=False)
    # my_analysis.plot_with_cuts(scaling='xs', to_pkl=False)
    # my_analysis.make_all_cutgroup_2dplots(to_pkl=False)
    # my_analysis.kinematics_printouts()
    # my_analysis.print_cutflow_latex_table()
    # file_utils.convert_pkl_to_root(conv_all=True)

    # import utils.plotting_utils as pu
    # pu.plot_1d_hist(x=[my_analysis.truth['MC_WZ_dilep_m_born'],
    #                    my_analysis.truth['MC_WZ_dilep_m_bare'],
    #                    my_analysis.truth['MC_WZ_dilep_m_dres']],
    #                 bins=(50, 1, 500), is_logbins=True, log_y=True,
    #                 x_label='dilepton mass [GeV]', filename='bornbaredres_overlay',
    #                 legend_label=['Born', 'bare', 'dressed'], to_file=True, legend=True, yerr='sumw2')
