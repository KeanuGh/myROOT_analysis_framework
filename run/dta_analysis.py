from src.analysis import Analysis

if __name__ == '__main__':
    datasets = {
        'wtaunu_dta_h': {
            'data_path': '/mnt/D/data/DTA_outputs/user.kghorban.Sh_2211_Wtaunu_H*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            'TTree_name': 'T_s1thv_NOMINAL',
            'year': '2015+2016',
            'lepton': 'tau',
            'dataset_type': 'dta',
            'label': r'Sherpa $W\rightarrow\tau\nu\rightarrow h$',
        },
        'wtaunu_dta_l': {
            'data_path': '/mnt/D/data/DTA_outputs/user.kghorban.Sh_2211_Wtaunu_L*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            'TTree_name': 'T_s1tlv_NOMINAL',
            'year': '2015+2016',
            'lepton': 'tau',
            'dataset_type': 'dta',
            'label': r'Sherpa $W\rightarrow\tau\nu\rightarrow l$',
        },

        # 'wtaunu_analysistop_peak': {
        #     'data_path': '/mnt/D/data/analysistop_out/mc16a/w*taunu/*.root',
        #     'cutfile_path': '../options/DTA_cuts/analysistop_peak.txt',
        #     'lepton': 'tau',
        #     'dataset_type': 'analysistop',
        #     'hard_cut': 'M_W',
        #     'label': r'Powheg $W^-\rightarrow\tau\nu\rightarrow\mu\nu$',
        # },
        # 'wtaunu_analysistop': {
        #     'data_path': '/mnt/D/data/analysistop_out/mc16a/w*taunu_*/*.root',
        #     'cutfile_path': '../options/DTA_cuts/analysistop.txt',
        #     'lepton': 'tau',
        #     'dataset_type': 'analysistop',
        #     'label': r'Powheg $W^-\rightarrow\tau\nu\rightarrow\mu\nu$',
        # },
    }

    my_analysis = Analysis(datasets, analysis_label='dta_analysis', force_rebuild=False, skip_verify_pkl=True,
                           log_level=10, log_out='both', timedatelog=True, separate_loggers=False)

    my_analysis.apply_cuts(truth=True)

    my_analysis.plot_hist('wtaunu_dta_h', 'rwCorr', bins=(30, 0, 2.5))
    my_analysis.plot_hist('wtaunu_dta_h', 'prwWeight', bins=(30, 0, 2.5))
    my_analysis.plot_hist('wtaunu_dta_h', 'weight', bins=(30, -2500, 2500), logy=True)
    my_analysis.plot_hist('wtaunu_dta_h', 'TauSF_LooseWP', bins=(30, 0, 2.5), logy=True)
    my_analysis.plot_hist('wtaunu_dta_h', 'TauSF_MediumWP', bins=(30, 0, 2.5), logy=True)
    my_analysis.plot_hist('wtaunu_dta_h', 'TauSF_TightWP', bins=(30, 0, 2.5), logy=True)
    # my_analysis.plot_hist('wtaunu_dta_h', 'Jet_btSF', bins=(30, 0, 1.5), logy=True)
    # my_analysis.plot_hist('wtaunu_dta_h', 'Jet_JVT', bins=(30, -0.15, 1.5), logy=True)
    my_analysis.plot_hist('wtaunu_dta_h', 'JVT_SF', bins=(30, 0, 2.5), logy=True)
    my_analysis.plot_hist('wtaunu_dta_h', 'FJVT_SF', bins=(30, 0, 1.5), logy=True)

    my_analysis['wtaunu_dta_h'].profile_plot('TruthMTW', 'rwCorr', logx=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthMTW', 'prwWeight', logx=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthMTW', 'weight', logx=True, logy=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthMTW', 'TauSF_LooseWP', logx=True, logy=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthMTW', 'TauSF_MediumWP', logx=True, logy=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthMTW', 'TauSF_TightWP', logx=True, logy=True)
    # my_analysis['wtaunu_dta_h'].profile_plot('TruthMTW', 'Jet_btSF', logx=True, logy=True)
    # my_analysis['wtaunu_dta_h'].profile_plot('TruthMTW', 'Jet_JVT', logx=True, logy=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthMTW', 'JVT_SF', logx=True, logy=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthMTW', 'FJVT_SF', logx=True, logy=True)

    my_analysis['wtaunu_dta_h'].profile_plot('TruthTauPt', 'rwCorr', logx=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthTauPt', 'prwWeight', logx=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthTauPt', 'weight', logx=True, logy=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthTauPt', 'TauSF_LooseWP', logx=True, logy=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthTauPt', 'TauSF_MediumWP', logx=True, logy=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthTauPt', 'TauSF_TightWP', logx=True, logy=True)
    # my_analysis['wtaunu_dta_h'].profile_plot('TruthTauPt', 'Jet_btSF', logx=True, logy=True)
    # my_analysis['wtaunu_dta_h'].profile_plot('TruthTauPt', 'Jet_JVT', logx=True, logy=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthTauPt', 'JVT_SF', logx=True, logy=True)
    my_analysis['wtaunu_dta_h'].profile_plot('TruthTauPt', 'FJVT_SF', logx=True, logy=True)

    # leptonic decay
    my_analysis.plot_hist('wtaunu_dta_l', 'rwCorr', bins=(30, 0, 2.5))
    my_analysis.plot_hist('wtaunu_dta_l', 'prwWeight', bins=(30, 0, 2.5))
    my_analysis.plot_hist('wtaunu_dta_l', 'weight', bins=(30, -2500, 2500), logy=True)
    my_analysis.plot_hist('wtaunu_dta_l', 'TauSF_LooseWP', bins=(30, 0, 2.5), logy=True)
    my_analysis.plot_hist('wtaunu_dta_l', 'TauSF_MediumWP', bins=(30, 0, 2.5), logy=True)
    my_analysis.plot_hist('wtaunu_dta_l', 'TauSF_TightWP', bins=(30, 0, 2.5), logy=True)
    # my_analysis.plot_hist('wtaunu_dta_l', 'Jet_btSF', bins=(30, 0, 1.5), logy=True)
    # my_analysis.plot_hist('wtaunu_dta_l', 'Jet_JVT', bins=(30, -0.15, 1.5), logy=True)
    my_analysis.plot_hist('wtaunu_dta_l', 'JVT_SF', bins=(30, 0, 2.5), logy=True)
    my_analysis.plot_hist('wtaunu_dta_l', 'FJVT_SF', bins=(30, 0, 1.5), logy=True)

    my_analysis['wtaunu_dta_l'].profile_plot('TruthMTW', 'rwCorr', logx=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthMTW', 'prwWeight', logx=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthMTW', 'weight', logx=True, logy=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthMTW', 'TauSF_LooseWP', logx=True, logy=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthMTW', 'TauSF_MediumWP', logx=True, logy=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthMTW', 'TauSF_TightWP', logx=True, logy=True)
    # my_analysis['wtaunu_dta_l'].profile_plot('TruthMTW', 'Jet_btSF', logx=True, logy=True)
    # my_analysis['wtaunu_dta_l'].profile_plot('TruthMTW', 'Jet_JVT', logx=True, logy=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthMTW', 'JVT_SF', logx=True, logy=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthMTW', 'FJVT_SF', logx=True, logy=True)

    my_analysis['wtaunu_dta_l'].profile_plot('TruthTauPt', 'rwCorr', logx=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthTauPt', 'prwWeight', logx=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthTauPt', 'weight', logx=True, logy=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthTauPt', 'TauSF_LooseWP', logx=True, logy=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthTauPt', 'TauSF_MediumWP', logx=True, logy=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthTauPt', 'TauSF_TightWP', logx=True, logy=True)
    # my_analysis['wtaunu_dta_l'].profile_plot('TruthTauPt', 'Jet_btSF', logx=True, logy=True)
    # my_analysis['wtaunu_dta_l'].profile_plot('TruthTauPt', 'Jet_JVT', logx=True, logy=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthTauPt', 'JVT_SF', logx=True, logy=True)
    my_analysis['wtaunu_dta_l'].profile_plot('TruthTauPt', 'FJVT_SF', logx=True, logy=True)

    # my_analysis.merge_datasets('wtaunu_analysistop', 'wtaunu_analysistop_peak')

    # my_analysis.apply_cuts()

    # my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthTauPt', 'MC_WZmu_el_pt_dres'],
    #                       xlabel=r'Tau $p_T$ [GeV]', filename='taupt_cuts.png', bins=(30, 1, 2000),
    #                       ratio_axlim=1.5, weight='truth_weight', normalise=True, stats_box=True, ratio_fit=True)
