#exec(open('confidence_intervals.py').read())
import subprocess as sp
import numpy as np
import pandas as pd
import importlib as il
import matplotlib.pyplot as plt
import math
import plots

if __name__ == '__main__':
    sp.call('cls', shell = True)
    il.reload(plots)
    plt.close()

    # ----------------------------------------------------------------------
    # Create a population distribution for experimental purposes.
    # ----------------------------------------------------------------------
    mu = 175
    sigma = 15
    population = np.random.randn(1000000) * sigma + mu;
    mu = np.mean(population)                # population mean
    sigma = np.std(population, ddof = 0)    # population standard deviation
    print('Population Characteristics:')
    print('Population mean = ' + str(round(mu, 2)))
    print('Population standard deviation = ' + str(round(sigma, 2)))
    print('Population range = [' +
           str(round(min(population), 2)) + ',' +
           str(round(max(population), 2)) + ']')
    print('')
    dfPopulation = pd.DataFrame(population, columns = ['height'])

    # ----------------------------------------------------------------------
    # Draw a sample from from the population with sample size of n.
    # ----------------------------------------------------------------------
    n = 20
    sample = np.random.choice(population, size = n)
    dfSample = pd.DataFrame(sample, columns = ['height'])
    xbar = round(np.mean(sample), 2)        # sample mean
    sx = round(np.std(sample, ddof = 0), 2) # sample standard deviation
    print('Single Sample:')
    print('Sample mean = ' + str(round(xbar, 2)))
    print('Sample standard deviation = ' + str(round(sx, 2)))
    print('')

    # create the sampling distribution
    lsx = list()
    for cnt in range(0, 100000):
        samp = np.random.choice(population, size = n)
        lsx.append(np.mean(samp))
    dfSamplingDist = pd.DataFrame(lsx, columns = ['height'])
    x = np.array(lsx)
    x.sort()
    sampmu = x.mean()
    sampsigma = sigma / math.sqrt(n)
    ex = round(x.mean(), 2)
    sex = round(np.std(x), 2)
    print('Sampling Distribution:')
    print('E(xbar) = {0:.8}'.format(ex))
    print('SE(xbar) = {0:.8}'.format(sex))
    print('')

    # ----------------------------------------------------------------------
    # Plot a histogram of the sampling distribution and the probability density
    # function for the corrensponding normal distribution. Compare the shape
    # of the histogram and the corresponding normal distribution with variance
    # sigma / sqrt(n). Note that their shapes are identical.
    # ----------------------------------------------------------------------
    nrow = 2
    ncol = 1
    nplot = 1
    fig = plt.figure(figsize = (14.4, 9))
    ax = fig.add_subplot(nrow, ncol, nplot)
    plots.histogram(dfSamplingDist
        ,fig = fig
        ,ax = ax
        ,numBins = 20
        ,title = 'Histogram vs Normal Distribution'
        ,xlabel = ['height']
        ,ylabel = ['count'])
    nplot = nplot + 1

    # normal distribution centered around sample distribution mean with variance
    # sigma / sqrt(n)
    fx = 1 / (math.sqrt(2 * math.pi) * sampsigma) * np.exp(-((x - sampmu)**2) / (2 * sampsigma**2))
    plots.scatter(x
        ,fx
        ,fig = fig
        ,ylim = (fx.min(), fx.max())
        ,title = ''
        ,axesNew = True
        ,markersize = 0
        ,linewidth = 2
        ,color = (1, 0.4, 0.4, 1))

    # ----------------------------------------------------------------------
    # Plot the corrensponding normal distributions of the population and sampling
    # distributions. Note that the sampling distribution is much "tighter" implying
    # a smaller variance. Note that both are centered around the same value.
    # ----------------------------------------------------------------------
    ax = fig.add_subplot(nrow, ncol, nplot)
    plots.scatter(x
        ,fx
        ,fig = fig
        ,ax = ax
        ,ylim = (fx.min(), fx.max())
        ,title = ''
        ,markersize = 0
        ,linewidth = 2
        ,color = (1, 0.4, 0.4, 1))
    x = dfPopulation.loc[:, 'height'].values
    xmin = x.min()
    xmax = x.max()
    x = np.linspace(xmin, xmax, 100)
    fx = 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((x - mu)**2) / (2 * sigma**2))
    ylim = ax.get_ylim()
    if fx.max() > ylim[1]:
        ylim = (ylim[0], ylim[1])
    plots.scatter(x
        ,fx
        ,fig = fig
        ,ax = ax
        ,ylim = ylim
        ,title = ''
        ,markersize = 0
        ,linewidth = 2
        ,color = (0.4, 0.4, 1, 1))
    nplot = nplot + 1
    ax.legend(['Sampling Distribution','Population Distribution'])
    ax.set_xlabel('height')
    ax.set_ylabel('f(x)')

    # TODO: code based on quickstart.ipynb





    fig.tight_layout()
    plt.show()
