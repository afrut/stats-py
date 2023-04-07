#exec(open('scratch.py').read())
import subprocess as sp
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import importlib as il
import pandas as pd

import plots

if __name__ == '__main__':
    sp.call('cls', shell = True)
    plt.close('all')
    il.reload(plots)

    samp1 = np.array([275, 286, 287, 271, 283, 271, 279, 275, 263, 267])
    samp2 = np.array([258, 244, 260, 265, 273, 281, 271, 270, 263, 268])

    n1 = len(samp1)
    n2 = len(samp2)
    x1 = samp1.mean()
    x2 = samp2.mean()
    s1 = samp1.std(ddof = 1)
    s2 = samp2.std(ddof = 1)

    df1 = pd.DataFrame(samp1, columns = ['d'])
    df2 = pd.DataFrame(samp2, columns = ['d'])
    df3 = pd.DataFrame(np.concatenate([samp1, samp2], axis = 0), columns = ['d'])
    plots.probplot(df1, title = 'Brand 1')
    plots.probplot(df2, title = 'Brand 2')
    plots.probplot(df3, title = 'All')

    plt.show()
