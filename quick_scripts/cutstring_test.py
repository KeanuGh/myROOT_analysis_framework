from dataclasses import dataclass
import pandas as pd
import numpy as np
import operator as op
from src.histogram import Histogram1D
from time import time


@dataclass
class Cut:
    name: str
    cutstr: str
    is_symmetric: bool
    tree: str


op_dict = {
    '<': op.lt,
    '<=': op.le,
    '=': op.eq,
    '!=': op.ne,
    '>': op.gt,
    '>=': op.ge,
}


df = pd.DataFrame({'eta1': 3*np.random.random(1000), 'eta2': 3*np.random.random(1000)})
cutstr1 = 'eta1.abs() < 1.37 or eta1.abs() > 1.57'
cutstr2 = 'eta1 < 2.47'


def cut_on_cutstr(data: pd.DataFrame, cutstr: str) -> pd.DataFrame:
    t = time()
    data = data.query(cutstr)
    print(f"{cutstr}: {time() - t:.2g}")
    return data


bins = (20, 0, 3)
h_all = Histogram1D(df['eta1'], bins)
h1 = Histogram1D(cut_on_cutstr(df, cutstr1)['eta1'], bins)
h2 = Histogram1D(cut_on_cutstr(df, cutstr2)['eta1'], bins)
# h3 = Histogram1D(df.loc[df['eta1'] < 1 & df['eta1'] > 2], bins)

h_all.plot(yerr=False, show=True, label='normal')
h1.plot(yerr=False, show=True, label=cutstr1)
h2.plot(yerr=False, show=True, label=cutstr2)
# h3.plot(yerr=False, show=True, label='manual')
