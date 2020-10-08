#exec(open('distributions\\stdnorm.py').read())
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
import counting_techniques as ct
import math

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
    # Standard Normal Distribution
    # The Standard Normal Distribution is simply the Normal Distribution
    # centered around 0 and has a standard deviation of 1. A random variable
    # that is distribution according to this distribution is denoted Z and its
    # values are called z-values. Standardization of a normal variable X
    # results in a z-value. z-value = (x - mu) / sigma.
    #
    # ----------------------------------------------------------------------
    print('----------------------------------------')
    print('  Standard Normal Distribution')
    print('----------------------------------------')
    # ----------------------------------------
    # probability density function
    # ----------------------------------------
    mu = 0
    sigma = 1
    x = 1.5
    print('{0}A random variable Z is standard-normally distributed'.format(space))
    prob = stats.norm.pdf(x, mu, sigma)
    prob2 = 1/(math.sqrt(2 * math.pi) * sigma) * math.exp(-(x - mu)**2 * (1 / (2 * sigma**2)))
    assert(round(prob - prob2, 8) == 0)
    print('{0}The probability of encountering x = {1} is {2:.8}.'\
        .format(space, x, prob))
    probs = list()
    xs = np.arange(mu - 5, mu + 5 + 1, 1)
    for xl in xs:
        prob = stats.norm.pdf(xl, mu, sigma)
        probs.append(prob)
        print('{0}The probability of encountering x = {1} is {2:.8}.'\
            .format(space, xl, prob))

    # ----------------------------------------
    # plotting
    # ----------------------------------------
    xs = np.array(xs)
    probs = np.array(probs).round(4)
    plots.barplot(xs, probs
        ,title = 'Standard Normal Distribution; mu = {0}, sigma = {1}'.format(mu, sigma)
        ,align = 'edge'
        ,edgecolor = edgecolor
        ,show = False, close = False)
    print('')

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    x = 0
    h = 1e-6
    probcum = stats.norm.cdf(x, mu, sigma)
    xs = np.arange((int)(-6 * sigma), x, h)
    probs = stats.norm.pdf(xs, mu, sigma)
    probcum2 = probs * h
    probcum2 = probcum2.sum()
    assert(round(abs(probcum - probcum2), 4) == 0)
    print('{0}The probability that x <= {1} P(x <= {2}) = {3:.8}'.format(space, x, x, probcum))






    # show plots
    if(show):
        plt.show()
    else:
        plt.close()
