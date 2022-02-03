import os
import time
from multiprocessing.pool import ThreadPool as Pool

import ROOT
import matplotlib.pyplot as plt
import numpy as np

from src.histogram import Histogram1D

nbins = 20
xmin = -1
xmax = 1

boost_unthreaded_times = []
boost_threaded_times = []
root_loop_times = []
root_map_times = []
root_comp_times = []
root_pool_times = []
x = np.logspace(1, 8, 8, dtype=int)
for n in x:
	print("n = ", n)
	a = np.random.normal(size=n)
	w = np.random.random(n)

	hR = ROOT.TH1F('test', 'test;x;entries', nbins, xmin, xmax)
	hH = Histogram1D(bins=(nbins, xmin, xmax))

	t0 = time.time()
	hH.fill(a, weight=w)
	boost_unthreaded_times.append(time.time() - t0)

	t1 = time.time()
	hH.fill(a, weight=w, threads=0)
	boost_threaded_times.append(time.time() - t1)

	t2 = time.time()
	for ia, iw in zip(a, w):
		hR.Fill(ia, iw)
	root_loop_times.append(time.time() - t2)

	t3 = time.time()
	map(hR.Fill, *zip(a, w))
	root_map_times.append(time.time() - t3)

	t4 = time.time()
	{hR.Fill(ia, iw) for ia, iw in zip(a, w)}
	root_comp_times.append(time.time() - t4)

	t5 = time.time()
	with Pool(os.cpu_count()) as p:
		p.map(hR.Fill, a)
	root_pool_times.append(time.time() - t5)

print('plotting...')
plt.plot(x, boost_unthreaded_times, label='boost-histogram - unthreaded')
plt.plot(x, boost_threaded_times, label='boost-histogram - threaded')
plt.plot(x, root_loop_times, label='pyROOT - for loop')
plt.plot(x, root_map_times, label='pyROOT - map')
plt.plot(x, root_comp_times, label='pyROOT - comprehension')
plt.plot(x, root_pool_times, label='pyROOT - thread pool')
plt.semilogx()
plt.semilogy()
plt.legend()
plt.xlabel('array length')
plt.ylabel('time to fill (s)')
plt.show()
