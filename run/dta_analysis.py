from src.analysis import Analysis

if __name__ == '__main__':
    datasets = {
        'wtaunu_dta': {
            'data_path': '/mnt/D/data/DTA_outputs/user.kghorban.Sh_2211_Wtaunu_L*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            'TTree_name': 'T_s1tlv_NOMINAL',
            'year': '2015+2016',
            'lepton': 'tau',
            'dataset_type': 'dta',
            'label': r'Sherpa $W\rightarrow\tau\nu\rightarrow h$',
        },

        'wtaunu_analysistop_peak': {
            'data_path': '/mnt/D/data/analysistop_out/mc16a/w*taunu/*.root',
            'cutfile_path': '../options/DTA_cuts/analysistop.txt',
            'lepton': 'tau',
            'dataset_type': 'analysistop',
            'label': r'Powheg $W^-\rightarrow\tau\nu\rightarrow\mu\nu$',
        },
        'wtaunu_analysistop': {
            'data_path': '/mnt/D/data/analysistop_out/mc16a/w*taunu_*/*.root',
            'cutfile_path': '../options/joanna_cutflow/DY_HM.txt',
            'lepton': 'tau',
            'dataset_type': 'analysistop',
            'label': r'Powheg $W^-\rightarrow\tau\nu\rightarrow\mu\nu$',
        },
    }

    my_analysis = Analysis(datasets, analysis_label='dta_analysis', force_rebuild=True,
                           log_level=10, log_out='console', timedatelog=True, separate_loggers=False)

    my_analysis.merge_datasets('wtaunu_analysistop', 'wtaunu_analysistop_peak')

    my_analysis.apply_cuts()

    my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthTauPt', 'MC_WZmu_el_pt_born'],
                          xlabel=r'Tau $p_T$ [GeV]', filename='taupt_cuts.png', bins=(30, 1, 2000),
                          ratio_axlim=1.5, weight='truth_weight', normalise=True, stats_box=True, ratio_fit=True)
