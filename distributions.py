#exec(open('distributions.py').read())
import subprocess as sp
import importlib as il
import numpy as np
import scipy.stats as stats
import plots
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sp.call('cls', shell = True)
    il.reload(plots)
    plt.close()
    space = '    '

    # ----------------------------------------------------------------------
    # Plotting Parameters
    # ----------------------------------------------------------------------
    edgecolor = np.array([0.121568627,0.466666667,0.705882353]) / 1.6

    # ----------------------------------------------------------------------
    #
    # Binomial Distribution
    # Consider the following situation: The probability of encountering a
    # poisonous molecule in a sample is 0.1. What is the probability of 5
    # samples containing a poisonous molecule in the next 20 samples? Take an
    # event that has a probability of occurrence p. The binomial distribution
    # gives the probability of this event occurring k times in the next n
    # events.
    #
    # ----------------------------------------------------------------------
    p = 0.1
    k = 2
    n = 18
    prob = stats.binom.pmf(k, n, p)
    print('An event E occurs with a probability of {0}.'.format(p))
    print('{0} The probability that E occurs {1} times'.format(space, k) +\
        ' in the next {0} events is {1:.4}.'.format(n, prob))
    x = list()
    probs = list()
    for k in range(0, n + 1):
        prob = stats.binom.pmf(k, n, p)
        x.append(k)
        probs.append(round(prob, 4))
        print('{0} The probability that E occurs {1} times'.format(space, k) +\
        ' in the next {0} events is {1:.4}.'.format(n, prob))

    k = 4
    prob = 0
    for k in range(0, k):
        prob = prob + stats.binom.pmf(k, n, p)
    prob = 1 - prob
    print('{0} The probability that E occurs >= {1} times is {2:.4}'.format(space, k + 1, prob))

    k1 = 3
    k2 = 7
    prob = 0
    for k in range(k1, k2):
        prob = prob + stats.binom.pmf(k, n, p)
    probcum = stats.binom.cdf(k2, n, p) - stats.binom.cdf(k1, n, p)\
        - stats.binom.pmf(k2, n, p) + stats.binom.pmf(k1, n, p)
    print('{0} The probability that E occurs {1} <= k < {2} times is {3:.4}'.format(space, k1, k2, prob))

    x = np.array(x)
    probs = np.array(probs)
    plots.barplot(x, probs
        ,title = 'Binomial Distribution; p = {0:.4}, n = {1}, k = {2}'.format(p, n, k)
        ,align = 'edge'
        ,edgecolor = edgecolor
        ,show = True, close = True)
