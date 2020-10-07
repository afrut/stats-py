#exec(open('distributions.py').read())
# ----------------------------------------------------------------------
#
# Reference Material Montgomery & Runger: Applied statistics and Probability
# for Engineers 7ed
#
# ----------------------------------------------------------------------
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
    edgecolor = np.array([0.121568627,0.466666667,0.705882353]) / 1.6   # color of the edges of the bar graph rectangles
    show = False                                                        # whether or not to show plots

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
    print('----------------------------------------')
    print('  Binomial Distribution')
    print('----------------------------------------')
    print('{0}An event E occurs with a probability of {1}.'.format(space, p))
    prob = stats.binom.pmf(k, n, p)
    assert(0.284 - round(prob,3) == 0)
    print('{0}The probability that E occurs {1} times'.format(space, k) +\
        ' in the next {0} events is {1:.4}.'.format(n, prob))
    x = list()
    probs = list()
    for k in range(0, n + 1):
        prob = stats.binom.pmf(k, n, p)
        x.append(k)
        probs.append(round(prob, 4))
        print('{0}The probability that E occurs {1} times'.format(space, k) +\
            ' in the next {0} events is {1:.4}.'.format(n, prob))

    k = 4
    prob = 0
    probcum = stats.binom.cdf(k - 1, n, p)
    probcum = 1 - probcum
    for k in range(0, k):
        prob = prob + stats.binom.pmf(k, n, p)
    prob = 1 - prob
    assert(round(abs(probcum - prob), 4) == 0)
    assert(abs(0.098 - round(probcum, 3)) == 0)
    print('{0}The probability that E occurs >= {1} times in {2} events {3:.4} is'\
        .format(space, k + 1, n, prob))

    k1 = 3
    k2 = 7
    prob = 0
    for k in range(k1, k2):
        prob = prob + stats.binom.pmf(k, n, p)
    probcum = stats.binom.cdf(k2, n, p) - stats.binom.cdf(k1, n, p)\
        - stats.binom.pmf(k2, n, p) + stats.binom.pmf(k1, n, p)
    assert(round(abs(prob - probcum), 4) == 0)
    print('{0}The probability that E occurs {1} <= k < {2} times in {3} events is {4:.4}'\
        .format(space, k1, k2, n, prob))

    mu = stats.binom.mean(n, p)
    sigmasq = stats.binom.var(n, p)
    assert(round(mu - (n * p), 6) == 0)
    assert(round(sigmasq - (n * p * (1 - p)), 6) == 0)
    print('{0}The mean of this binomial distribution is {1:.4}'.format(space, mu))
    print('{0}The variance of this binomial distribution is {1:.4}'.format(space, sigmasq))

    x = np.array(x)
    probs = np.array(probs)
    plots.barplot(x, probs
        ,title = 'Binomial Distribution; p = {0:.4}, n = {1}, k = {2}'.format(p, n, k)
        ,align = 'edge'
        ,edgecolor = edgecolor
        ,show = False, close = False)
    print('')

    # ----------------------------------------------------------------------
    #
    # Geometric Distribution
    # Consider the following situation: The probability of encountering a
    # poisonous molecule in a sample is 0.1. What is the probability of
    # encountering the first sample with a poisonous molecule on the 10th
    # sample? How about the 15th? The geometric distribution answers these
    # questions. It gives the probability that the first occurrence of event E
    # (with probability p) occurs in exactly k events.
    #
    # ----------------------------------------------------------------------
    p = 0.1
    k = 5
    print('----------------------------------------')
    print('  Geometric Distribution')
    print('----------------------------------------')
    print('{0}An event E occurs with a probability of {1}.'.format(space, p))
    prob = stats.geom.pmf(k, p)
    print('{0}The probability that E first occurs in {1} times is {2:.4}'\
        .format(space, k, prob))
    assert(0.066 == round(prob, 3))
    x = list()
    probs = list()
    ks = k + 1
    for k in range(0, ks):
        prob = stats.geom.pmf(k, p)
        x.append(k)
        probs.append(round(prob, 4))
        print('{0}The probability that E first occurs in {1} times is {2:.4}'\
            .format(space, k, prob))

    k = 4
    prob = 0
    for k in range(0, k):
        prob = prob + stats.geom.pmf(k, p)
    prob = 1 - prob
    assert(0.729 == round(prob, 3))
    print('{0}The probability that E first occurs in >= {1} events is {2:.4}'.format(space, k + 1, prob))

    k1 = 3
    k2 = 7
    prob = 0
    for k in range(k1, k2):
        prob = prob + stats.geom.pmf(k, p)
    probcum = stats.geom.cdf(k2, p) - stats.geom.cdf(k1, p)\
        - stats.geom.pmf(k2, p) + stats.geom.pmf(k1, p)
    assert(round(abs(probcum - prob), 4) == 0)
    print('{0}The probability that first E occurs in {1} <= k < {2} events is {3:.4}'.format(space, k1, k2, prob))

    mu = stats.geom.mean(p)
    sigmasq = stats.geom.var(p)
    assert(round(mu - (1 / p), 6) == 0)
    assert(round(sigmasq - ((1 - p) / p**2), 6) == 0)
    print('{0}The mean of this geometric distribution is {1:.4}'.format(space, mu))
    print('{0}The variance of this geometric distribution is {1:.4}'.format(space, sigmasq))

    x = np.array(x)
    probs = np.array(probs)
    plots.barplot(x, probs
        ,title = 'Geometric Distribution; p = {0:.4}, n = {1}, k = {2}'.format(p, n, k)
        ,align = 'edge'
        ,edgecolor = edgecolor
        ,show = False, close = False)
    print('')

    # ----------------------------------------------------------------------
    #
    # Negative Binomial Distribution
    # This is similar to the Geometric Distribution with a slight twist.
    #
    # Consider the following situation: The probability of encountering a
    # poisonous molecule in a sample is 0.1. What is the probability of
    # encountering the exactly 3 samples with a poisonous molecule by the 10th
    # sample? How about exactly 5 samples by the 15th? The Negative Binomial
    # Distribution answers these questions. It gives the probability that the
    # kth occurrence of event E (with probability p) occurs by exactly n events.
    #
    # ----------------------------------------------------------------------
    p = 0.1
    n = 10          # number of events
    k = 4           # number of occurrences of event E
    pplot = p
    nplot = n
    kplot = k
    print('----------------------------------------')
    print('  Negative Binomial Distribution')
    print('----------------------------------------')
    print('{0}An event E occurs with a probability of {1}.'.format(space, p))
    # pmf(num events - num event E, num event E, probability)
    prob = stats.nbinom.pmf(n - k, k, p)
    print('{0}The probability that E occurs exactly {1} times by the {2}th event is {3:.4}'\
        .format(space, k, n, prob))
    assert(round(abs(prob - 0.004464),6) == 0)
    x = list()
    probs = list()
    for k in range(1, n + 1):
        prob = stats.nbinom.pmf(n - k, k, p)
        x.append(k)
        probs.append(round(prob, 4))
        print('{0}The probability that E occurs exactly {1} times by the {2}th event is {3:.4}'\
            .format(space, k, n, prob))

    n = 5
    k = 3
    p  = 0.2
    print('')
    print('{0}An event E occurs with a probability of {1}.'.format(space, p))
    prob = 0
    ns = range(k, n + 1)
    for n in ns:
        prob = prob + stats.nbinom.pmf(n - k, k, p)
    probcum = stats.nbinom.cdf(n - k, k, p)
    assert(round(abs(prob - probcum), 6) == 0)
    assert(0.058 == round(probcum,3))
    print('{0}The probability that E occurs exactly {1} times by the <= {2}th event is {3:.4}'\
        .format(space, k, n, prob))

    mu = stats.nbinom.mean(k, p, k)
    sigmasq = stats.nbinom.var(k, p)
    assert(round(mu - (k / p), 6) == 0)
    assert(round(sigmasq - (k * (1 - p) / p**2), 6) == 0)
    print('{0}The mean of this negative binomial distribution is {1:.4}'.format(space, mu))
    print('{0}The variance of this negative binomial distribution is {1:.4}'.format(space, sigmasq))

    x = np.array(x)
    probs = np.array(probs)
    plots.barplot(x, probs
        ,title = 'Negative Binomial Distribution; p = {0:.4}, n = {1}, k = {2}'\
            .format(pplot, nplot, kplot)
        ,align = 'edge'
        ,edgecolor = edgecolor
        ,show = False, close = False)
    print('')

    # show plots
    if(show):
        plt.show()
    else:
        plt.close()
