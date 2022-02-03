import matplotlib.pyplot as plt
import numpy as np

from src.histogram import Histogram1D

a = np.random.normal(size=1000)
b = np.random.normal(size=1000)
w = np.random.random(1000)
w2 = np.random.random(1000)

h = Histogram1D(var=a, bins=(10, -1, 1), weight=w)
h2 = Histogram1D(var=a, bins=(10, -1, 1), weight=w)
ho1 = Histogram1D(var=b, bins=(10, -1, 1), weight=w)
ho2 = Histogram1D(var=b, bins=(10, -1, 1), weight=w2)

h.plot()
plt.title('normal')
plt.show()

h3 = h.copy()
h3.plot()
plt.title('copy h')
plt.show()

hnorm = h.normalised()
hnorm.plot()
plt.title('h normalised')
plt.show()

h.plot(normalise=True)
plt.title('normalised in plot')
plt.show()

h2 /= h2.integral
h2.plot()
plt.title('normalised outside plot')
plt.show()

h.plot_ratio(h2)
plt.title(f'ratio')
plt.show()

h.plot_ratio(h2, normalise=True)
plt.title(f'ratio normalised')
plt.show()

ho1.plot()
plt.title(f'other histogram same weights')
plt.show()

ho2.plot()
plt.title(f'other histogram different weights')
plt.show()

# h3 /= ho2
# h3.plot()
# plt.title('h copy / other histogram diffent weights')
# plt.show()

h.plot_ratio(ho1)
plt.title('h ratio plot other histogram same weights')
plt.show()

fig, ax = plt.subplots()
h.plot_ratio(ho2, ax=ax)
plt.title('h ratio plot other histogram different weights')
fig.show()
