#exec(open('distributions\\norm.py').read())
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
    # Normal Distribution
    # Take a sample of 10 similar objects, measure them, and average their
    # measurements and call this a1. Take another sample of 10, measure them,
    # average and call this a2. Take yet another sample of 10, measure,
    # average, and call this a3. Do this many, many times, create a histogram
    # of the sample averages and the distribution of these sample avergaes will
    # be approximately Normal. This is the Central Limit Theorem.
    #
    # ----------------------------------------------------------------------
    print('----------------------------------------')
    print('  Normal/Gaussian Distribution')
    print('----------------------------------------')
    # ----------------------------------------
    # probability density function
    # ----------------------------------------
    mu = (float)(10)
    sigma = (float)(4)
    x = 5
    print('{0}A random variable X is normally distributed'.format(space) +
        ' with mu = {0} and sigma = {1}.'.format(mu, sigma))
    prob = stats.norm.pdf(x, mu, sigma)
    prob2 = 1/(math.sqrt(2 * math.pi) * sigma) * math.exp(-(x - mu)**2 * (1 / (2 * sigma**2)))
    assert(round(prob - prob2, 8) == 0)
    print('{0}The probability of encountering x = {1} is {2:.8}.'\
        .format(space, x, prob))
    probs = list()
    xs = np.arange(mu - 5, mu + 5 + 1, (mu / 10))
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
        ,title = 'Normal Distribution; mu = {0:.8}, sigma = {1}'.format(mu, sigma)
        ,align = 'edge'
        ,edgecolor = edgecolor
        ,show = False, close = False)
    print('')

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    x = 7
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
