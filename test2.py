#exec(open('test2.py').read())
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

    ## 4
    #c = 2/285
    #val = (c/2) * ((243/5) + (243/2) + 81 + 7203 - 243)
    #print(val)

    ## 5
    #val = (0.7 * 0.1) + ((0.9)**2) - ((0.7)**2)
    #print(val)
    #val = ((0.7**3)/2) + ((2/3) * (1 - (0.7**3)))
    #print(val)

    ## 6
    #counts = np.array([0, 13, 10 + 3, 5 + 9, 3 + 8, 9, 2, 1])
    #tot = counts.sum()
    #tot = 13 + 10 + 5 + 3 + 3 + 9 + 8 + 9 + 2 + 1
    #cum = 13 + 10 + 3

    ## 7
    #tot = 61 + 140 + 342
    #cnt = 0.8656 * tot
    #cum = 140 + 61 + 130 + 139

    ## 8
    #mu = 163
    #sigma = 86
    #xhi = 179
    #prob = 1 - stats.norm.cdf(xhi, loc = mu, scale = sigma)
    #print(prob * 100)

    ## 9
    #sigma = 19
    #xhi = 70
    #p = 0.82
    #zhi = stats.norm.ppf(p)
    #mu = xhi - (zhi * sigma)
    #print(mu)

    ## 10
    #mu = 121.8
    #sigma = 2.8
    #n = 8
    #xlo = 120.7
    #xhi = 123.8
    #prob = stats.norm.cdf(xhi, loc = mu, scale = sigma/math.sqrt(n)) -\
    #    stats.norm.cdf(xlo, loc = mu, scale = sigma/math.sqrt(n))
    #print(prob)

    ## 11
    #x = 4.83
    #proba = 1 - ((1/65) * ((x**2) - 16))
    #print(proba)
    #xlo = 4
    #xhi = 4.29
    #probb = ((1/65) * (xhi**2 - 16)) - ((1/65) * (xlo**2 - 16))
    #print(probb)
    #ex = (2 / (65 * 3)) * ((9**3) - (4**3))
    #print(ex)
    #varx = (1 / (2*65) * (9**4 - 4**4)) - (ex**2)
    #print(varx)

    ## 13
    #c = 68
    #proba = c * (((1/(7 * 13)) * math.exp(-13/3)) - ((1/42)*math.exp(-6/3)) - ( (1/(7*13)) - (1/42) ))
    #print(proba)

    ## 14
    #c = 2 / 285
    #val = (c * (((1/4)*(3**4)) + (3**3))) + (2 * c * ((7**3) - (3**3)))
    #print(val)

    # 15
    mu = 4
    lamb = 1 / mu
    x = 8 * 60 / 106
    prob = stats.expon.cdf(x, 0, 1 / lamb)
    print(prob)

    lamb = 12.2
    zhi = (2 - lamb)/math.sqrt(lamb)
    prob = stats.norm.cdf(zhi)
    print(prob)
