import time

import uproot

data = '../data/mc16a_wmintaunu_SLICES/*.root'
branches = {'MC_WZmu_el_pt_born', 'MC_WZneutrino_pt_born', 'MC_WZmu_el_phi_born', 'MC_WZneutrino_phi_born'}
tree = 'truth'

t = time.time()
df = uproot.concatenate(data + ':' + tree, branches, library='pd', num_workers=16, begin_chunksize=1024)
print(f"took {time.time() - t}s")
