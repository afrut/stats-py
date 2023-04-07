#exec(open('misc.py').read())
import subprocess as sp
import numpy as np
import pandas as pd
import importlib as il
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import confidence_intervals as ci
import hypothesis_testing as ht
import plots

def f(x, mu, sigma):
    return (x + 0.5 - mu) / sigma

if __name__ == '__main__':
    sp.call('cls', shell = True)
    il.reload(plots)
    il.reload(ci)
    il.reload(ht)
    plt.close('all')

    samp = np.array([425, 431, 416, 419, 421, 436, 418, 410, 431, 433, 423, 426, 410, 435, 436, 428, 411, 426, 409, 437, 422, 428, 413, 416])
    samp.sort()
