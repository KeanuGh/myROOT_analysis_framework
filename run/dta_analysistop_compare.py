import numpy as np
from numpy import pi

from src.analysis import Analysis

# DTA_PATH = '/data/keanu/ditau_output/'
# ANALYSISTOP_PATH = '/data/atlas/HighMassDrellYan/mc16a'
# DATA_OUT_DIR = '/data/keanu/framework_outputs/'
DTA_PATH = '/mnt/D/data/DTA_outputs/2022-06-08/'
ANALYSISTOP_PATH = '/mnt/D/data/analysistop_out/mc16a/'
DATA_OUT_DIR = '/mnt/D/data/dataset_pkl_outputs/'


if __name__ == '__main__':
    datasets = {
        # dta w->taunu->munu
        'wtaunu_mu_dta': {
            'data_path': DTA_PATH + '/user.kghorban.Sh_2211_Wtaunu_L*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_tau_to_muons.txt',
            'TTree_name': 'T_s1tlv_NOMINAL',
            'hard_cut': 'Muonic Tau',
            'lepton': 'tau',
            'dataset_type': 'dta',
            # 'force_rebuild': True,
            # 'force_recalc_weights': True,
            'validate_duplicated_events': False,
            'label': r'Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \l\nu$',
        },
        # analysistop w->taunu->munu
        'wtaunu_analysistop': {
            'data_path': ANALYSISTOP_PATH + '/w*taunu_*/*.root',
            'cutfile_path': '../options/DTA_cuts/analysistop.txt',
            'lepton': 'tau',
            'dataset_type': 'analysistop',
            # 'force_rebuild': False,
            'label': r'Powheg $W\rightarrow\tau\nu\rightarrow \mu\nu$',
        },
        'wtaunu_analysistop_peak': {
            'data_path': ANALYSISTOP_PATH + '/w*taunu/*.root',
            'cutfile_path': '../options/DTA_cuts/analysistop_peak.txt',
            'lepton': 'tau',
            'dataset_type': 'analysistop',
            'hard_cut': 'M_W',
            # 'force_rebuild': True,
            'label': r'Powheg/Pythia 8 $W\rightarrow\tau\nu\rightarrow \mu\nu$',
        },
    }

    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year='2015+2016',
        # force_rebuild=True,
        analysis_label='dta_analysistop_compare',
        skip_verify_pkl=False,
        # force_recalc_cuts=True,
        log_level=10, log_out='both', timedatelog=True, separate_loggers=False)
    my_analysis.merge_datasets('wtaunu_analysistop', 'wtaunu_analysistop_peak')

    # HISTORGRAMS
    # ==================================================================================================================
    # look at monte carlo weights
    # my_analysis.plot_hist('wtaunu_analysistop', 'weight_mc', bins=(100, -5, 5), filename_suffix='_5', yerr=None, logy=True)
    # my_analysis.plot_hist('wtaunu_analysistop', 'weight_mc', bins=(100, -10000, 10000), filename_suffix='full', yerr=None, logy=True)
    # my_analysis.plot_hist('wtaunu_mu_dta', 'weight_mc', bins=(50, -1e6, 1e6), filename_suffix='_mill', logy=True, yerr=None)
    # my_analysis.plot_hist('wtaunu_mu_dta', 'weight_mc', bins=(50, -5e9, 5e9), logy=True, filename_suffix='_bill', yerr=None)

    # ratio plot arguments
    ratio_args = {
        # 'ratio_axlim': 1.5,
        'ratio_label': 'Powheg/Sherpa',
        'stats_box': True,
        'ratio_fit': False,
        'scale_by_bin_width': True,
    }
    mass_bins = np.array(
        [130, 140.3921, 151.6149, 163.7349, 176.8237, 190.9588, 206.2239, 222.7093, 240.5125, 259.7389, 280.5022,
         302.9253, 327.1409, 353.2922, 381.5341, 412.0336, 444.9712, 480.5419, 518.956, 560.4409, 605.242, 653.6246,
         705.8748, 762.3018, 823.2396, 889.0486, 960.1184, 1036.869, 1119.756, 1209.268, 1305.936, 1410.332, 1523.072,
         1644.825, 1776.311, 1918.308, 2071.656, 2237.263, 2416.107, 2609.249, 2817.83, 3043.085, 3286.347, 3549.055,
         3832.763, 4139.151, 4470.031, 4827.361, 5213.257])

    # my_analysis.apply_cuts(truth=True)

    # my_analysis.plot_hist('wtaunu_analysistop', 'mt_born', weight='truth_weight', bins=mass_bins, logx=True, logy=True, logbins=True)
    # my_analysis.plot_hist('wtaunu_analysistop', 'MC_WZ_dilep_m_born', weight='truth_weight', bins=mass_bins, logx=True, logy=True, logbins=True)
    # my_analysis.plot_hist('wtaunu_mu_dta', 'TruthMTW', weight='truth_weight', bins=mass_bins, logx=True, logy=True, logbins=True)
    #
    # my_analysis['wtaunu_analysistop'].plot_dsid('mt_born', weight='truth_weight', bins=mass_bins, logx=True, logy=True, logbins=True)
    # my_analysis['wtaunu_analysistop'].plot_dsid('MC_WZ_dilep_m_born', weight='truth_weight', bins=mass_bins, logx=True, logy=True, logbins=True)
    # my_analysis['wtaunu_mu_dta'].plot_dsid('TruthMTW', weight='truth_weight', bins=mass_bins, logx=True, logy=True, logbins=True)

    # calc weights
    # my_analysis.logger.info("Calculating DTA weights...")
    # for dsid, dsid_df in my_analysis['wtaunu_mu_dta'].df.groupby(level='DSID'):
    #     my_analysis.logger.debug(f"DSID {dsid}..")
    #     xs = PMG_tool.get_crossSection(dsid)
    #     kFactor = PMG_tool.get_kFactor(dsid)
    #     filterEfficiency = PMG_tool.get_genFiltEff(dsid)
    #     PMG_factor = xs * kFactor * filterEfficiency
    #     # sumw = dsid_df['mcWeight'].sum()
    #     sumw = ROOT_utils.get_dta_sumw(DTA_PATH + '/user.kghorban.Sh_2211_Wtaunu_L*/*.root')
    #     my_analysis.logger.debug(f"DSID {dsid}.. pmg: {PMG_factor}, lumi: {my_analysis['wtaunu_mu_dta'].lumi}, sum: {sumw}")
    #
    #     my_analysis['wtaunu_mu_dta'].df.loc[dsid, 'truth_weight'] = dsid_df['mcWeight'] * my_analysis['wtaunu_mu_dta'].lumi * \
    #                                                                 dsid_df['rwCorr'] * dsid_df['prwWeight'] * PMG_factor / sumw

    # TRUTH
    # -----------------------------------
    # unnormalised
    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthBosonM', 'MC_WZ_dilep_m_born'],
                          bins=mass_bins, weight='truth_weight', ratio_axlim=1.5,
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthTauPt', 'MC_WZmu_el_pt_born'],
                          bins=mass_bins, weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthTauEta', 'MC_WZmu_el_eta_born'],
                          bins=(30, -5, 5), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthTauPhi', 'MC_WZmu_el_phi_born'],
                          bins=(30, -pi, pi), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoPt', 'MC_WZneutrino_pt_born'],
                          bins=mass_bins, weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoEta', 'MC_WZneutrino_eta_born'],
                          bins=(30, -5, 5), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoPhi', 'MC_WZneutrino_phi_born'],
                          bins=(30, -pi, pi), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthMTW', 'mt_born'],
                          bins=mass_bins, weight='truth_weight', ratio_axlim=1.5,
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist('wtaunu_mu_dta', 'TruthMuonPt', weight='truth_weight', bins=mass_bins, logx=True, logy=True, logbins=True)
    my_analysis.plot_hist('wtaunu_mu_dta', 'TruthMuonEta', weight='truth_weight', bins=(30, -5, 5), logy=True)
    my_analysis.plot_hist('wtaunu_mu_dta', 'TruthMuonPhi', weight='truth_weight', bins=(30, -pi, pi), logy=True)

    # # # normalised
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthTauPt', 'MC_WZmu_el_pt_born'],
    #                       bins=mass_bins, weight='truth_weight',
    #                       title='normalised to 1', normalise=True, logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthTauEta', 'MC_WZmu_el_eta_born'],
    #                       bins=(30, -5, 5), weight='truth_weight',
    #                       title='normalised to 1', normalise=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthTauPhi', 'MC_WZmu_el_phi_born'],
    #                       bins=(30, -pi, pi), weight='truth_weight',
    #                       title='normalised to 1', normalise=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoPt', 'MC_WZneutrino_pt_born'],
    #                       bins=mass_bins, weight='truth_weight',
    #                       title='normalised to 1', normalise=True, logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoEta', 'MC_WZneutrino_eta_born'],
    #                       bins=(30, -5, 5), weight='truth_weight',
    #                       title='normalised to 1', normalise=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoPhi', 'MC_WZneutrino_phi_born'],
    #                       bins=(30, -pi, pi), weight='truth_weight',
    #                       title='normalised to 1', normalise=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthMTW', 'mt_born'],
    #                       bins=mass_bins, weight='truth_weight',
    #                       title='normalised to 1', normalise=True, logx=True, logbins=True, **ratio_args)

    # # RECO
    # # -----------------------------------
    # # calculate specific weights
    # my_analysis.logger.info("Calculating weights...")
    # my_analysis['wtaunu_mu_dta']['muon_reco_weight'] =   my_analysis['wtaunu_mu_dta']['reco_weight'] \
    #                                                    * my_analysis['wtaunu_mu_dta']['Muon_recoSF'] \
    #                                                    * my_analysis['wtaunu_mu_dta']['Muon_isoSF'] \
    #                                                    * my_analysis['wtaunu_mu_dta']['Muon_ttvaSF']
    # my_analysis['wtaunu_mu_dta'].dropna('muon_reco_weight', drop_inf=True)
    #
    # # un-normalised
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MuonPt', 'mu_pt'],
    #                       bins=mass_bins, weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MuonEta', 'mu_eta'],
    #                       bins=(30, -5, 5), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MuonPhi', 'mu_phi'],
    #                       bins=(30, -pi, pi), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MET_met', 'met_met'],
    #                       bins=mass_bins, weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MET_phi', 'met_phi'],
    #                       bins=(30, -pi, pi), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MTW', 'mu_mt_reco'],
    #                       bins=mass_bins, weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['Muon_d0sig', 'mu_d0sig'],
    #                       bins=(30, -3.5, 3.5), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['Muon_delta_z0_sintheta', 'mu_delta_z0_sintheta'],
    #                       bins=(30, -1, 1), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', **ratio_args)
    #
    # # normalised
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MuonPt', 'mu_pt'],
    #                       bins=mass_bins, weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MuonEta', 'mu_eta'],
    #                       bins=(30, -5, 5), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MuonPhi', 'mu_phi'],
    #                       bins=(30, -pi, pi), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MET_met', 'met_met'],
    #                       bins=mass_bins, weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MET_phi', 'met_phi'],
    #                       bins=(30, -pi, pi), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MTW', 'mu_mt_reco'],
    #                       bins=mass_bins, weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['Muon_d0sig', 'mu_d0sig'],
    #                       bins=(30, -3.5, 3.5), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['Muon_delta_z0_sintheta', 'mu_delta_z0_sintheta'],
    #                       bins=(30, -1, 1), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', **ratio_args)

    my_analysis.logger.info("DONE.")
