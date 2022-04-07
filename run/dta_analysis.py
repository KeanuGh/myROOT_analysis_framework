from src.analysis import Analysis

if __name__ == '__main__':
    datasets = {
        'wtaunu': {
            'data_path': '/mnt/D/data/DTA_outputs/user.kghorban.Sh_2211_Wtaunu_H_maxHTpTV2_CVetoBVeto.MC16a.v1.2022-04-01_histograms.root/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            'TTree_name': 'T_s1thv_NOMINAL',
            'year': '2015+2016',
            'lepton': 'tau',
            'label': r'$W\rightarrow\tau\nu\rightarrow h$',
        },
    }

    my_analysis = Analysis(datasets, analysis_label='dta_analysis', force_rebuild=False,
                           log_level=10, log_out='both', timedatelog=False, separate_loggers=False)

    my_analysis.plot_hist('wtaunu', 'TauPt', bins=(30, -5, 5), weight='reco_weight',
                          normalise=False, stats_box=True)
