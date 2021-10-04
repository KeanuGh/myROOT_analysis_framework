from time import time

import os
import pandas as pd
import psutil
import uproot as upr


def mem_footprint_MB():
    mem = psutil.Process(os.getpid()).memory_info().rss
    return mem/1024**2


root_files = '../../data/mc16d_wmintaunu/*'

branches = [
    'MC_WZmu_el_eta_born',
    'MC_WZmu_el_pt_born',
    'MC_WZneutrino_pt_born',
]
truth_trees = upr.pandas.iterate(root_files, treepath='truth', branches=branches)
print(f"memory footprint after loading iterator: {mem_footprint_MB():.2f} MB")

lsi = time()
list_data = [data for data in truth_trees]
lsf = time()
print(f"time to generate list: {lsf-lsi:.2f} s")
print(f"memory footprint from list: {mem_footprint_MB():.2f} MB")

pdi = time()
df = pd.concat(list_data)
pdf = time()
print(f"time to gen df {pdf-pdi:.3f} s")
print(f"memory footprint: {mem_footprint_MB(): .2f} MB")
print(df.info())

