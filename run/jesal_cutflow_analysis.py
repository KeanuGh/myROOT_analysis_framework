#!/users/keanu/miniconda3/envs/root_env/bin/python

from src.analysis import Analysis

if __name__ == '__main__':

    # dataset inputs
    datasets = {
        # 'wmintaunu_slices': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wmintaunu_*/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': True,
        #     'lepton': 'tau'
        # },
        # 'wmintaunu_inclusive': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wmintaunu/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        #     'lepton': 'tau'
        # },
        # 'wplustaunu_slices': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wplustaunu_*/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': True,
        #     'lepton': 'tau'
        # },
        # 'wplustaunu_inclusive': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wplustaunu/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        #     'lepton': 'tau'
        # },
        'wminmunu_slices': {
            'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wminmunu_*/*.root',
            'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
            'TTree_name': 'nominal_Loose',
            'is_slices': True,
            'lepton': 'muon'
        },
        # 'wminmunu_inclusive': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wminmunu/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        #     'lepton': 'muon'
        # },
        # 'wplusmunu_slices': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wplusmunu_*/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': True,
        #     'lepton': 'muon'
        # },
        # 'wplusmunu_inclusive': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wplusmunu/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        #     'lepton': 'muon'
        # },
        # 'single_antitop_s': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/single_antitop_s/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'single_antitop_t': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/single_antitop_t/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'single_top_s': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/single_top_s/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'single_top_t': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/single_top_t/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'ttbar': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/single_top_s/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'wt_antitop': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wt_antitop/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'wt_top': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wt_top/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'wwplnumqq': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wwplnumqq/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'wwpqqmlnu': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wwpqqmlnu/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'wzlnull': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wzlnull/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'wzlnununu': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wzlnununu/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'wzlnuqq': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wzlnuqq/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'wzqqll': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/wzqqll/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'zmunu_inclusive': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/zmunu/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'zmunu_slices': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/zmunu_*/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': True,
        # },
        # 'ztautau': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/ztautau/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'ztautau_slices': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/ztautau_*/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': True,
        # },
        # 'zzllll': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/zzllll/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'zzllnunu': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/zzllnunu/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
        # 'zzqqll': {
        #     'datapath': '/data/atlas/HighMassDrellYan/test_mc16a/zzqqll/*.root',
        #     'cutfile': 'options/jesal_cutflow/cutfile_jesal.txt',
        #     'TTree_name': 'nominal_Loose',
        #     'is_slices': False,
        # },
    }

    my_analysis = Analysis(datasets, analysis_label='jesal_cutflow',
                           force_rebuild=False, log_level=10, log_out='console')

    my_analysis.plot_1d(x='mu_mt', bins=(50, 120, 4000), title=r'$W\rightarrow\tau\nu$ (13 TeV)',
                        scaling='xs', log_y=True)
    # pipeline
    # my_analysis.plot_mass_slices(ds_name='wmintaunu_slices', xvar='MC_WZ_m',
    #                              inclusive_dataset='wmintaunu_inclusive', logx=True, to_pkl=True)
    # my_analysis.plot_mass_slices(ds_name='wplustaunu_slices', xvar='MC_WZ_m',
    #                              inclusive_dataset='wplustaunu_inclusive', logx=True, to_pkl=True)
    # my_analysis.plot_mass_slices(ds_name='wminmunu_slices', xvar='MC_WZ_m',
    #                              inclusive_dataset='wminmunu_inclusive', logx=True, to_pkl=True)
    # my_analysis.plot_mass_slices(ds_name='wplusmunu_slices', xvar='MC_WZ_m',
    #                              inclusive_dataset='wplusmunu_inclusive', logx=True, to_pkl=True)
    # my_analysis.convert_pkl_to_root()

    # my_analysis.gen_cutflow_hist(event=True)
    # my_analysis.plot_with_cuts(scaling='xs', to_pkl=False)
    # my_analysis.make_all_cutgroup_2dplots(to_pkl=False)
    # my_analysis.kinematics_printouts()
    # my_analysis.print_cutflow_latex_table()

    # import utils.plotting_utils as pu
    # pu.plot_1d_hist(x=[my_analysis.truth['MC_WZ_dilep_m_born'],
    #                    my_analysis.truth['MC_WZ_dilep_m_bare'],
    #                    my_analysis.truth['MC_WZ_dilep_m_dres']],
    #                 bins=(50, 1, 500), is_logbins=True, log_y=True,
    #                 x_label='dilepton mass [GeV]', filename='bornbaredres_overlay',
    #                 legend_label=['Born', 'bare', 'dressed'], to_file=True, legend=True, yerr='sumw2')
