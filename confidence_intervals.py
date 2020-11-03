#exec(open('confidence_intervals.py').read())
import subprocess as sp
import numpy as np
import pandas as pd
import importlib as il
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import plots

def pdfnorm(x, mu, sigma):
    return 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((x - mu)**2) / (2 * sigma**2))

if __name__ == '__main__':
    sp.call('cls', shell = True)
    il.reload(plots)
    plt.close('all')

    # ----------------------------------------------------------------------
    # Create a population of 1 million heights in cm with known mean and
    # variance. This data set will be used for simulations.
    # ----------------------------------------------------------------------
    mu = 175
    sigma = 5
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
    dfPop = pd.DataFrame(population, columns = ['height'])

    # ----------------------------------------------------------------------
    # Create the sampling distribution
    # ----------------------------------------------------------------------
    lsx = list()
    n = 20
    for cnt in range(0, 100000):
        samp = np.random.choice(population, size = n)
        lsx.append(np.mean(samp))
    dfSampDist = pd.DataFrame(lsx, columns = ['height'])
    x = np.array(lsx)
    x.sort()
    sampDistMu = x.mean()
    sampDistSigma = sigma / math.sqrt(n)
    ex = round(x.mean(), 2)
    sex = round(np.std(x), 2)
    print('Sampling Distribution:')
    print('E(sampMean) = {0:.8}'.format(ex))
    print('SE(sampMean) = {0:.8}'.format(sex))
    print('')

    figsize = (14.4, 9)

    # ----------------------------------------------------------------------
    # Plot the corrensponding normal distributions of the population and sampling
    # distributions. NOTE: The sampling distribution is much "tighter" implying
    # a smaller variance. Note that both are centered around the same value.
    # ----------------------------------------------------------------------
    # get min/max values of population and create an array of x-values
    x = dfPop.loc[:, 'height'].values
    xmin = x.min()
    xmax = x.max()
    x = np.linspace(xmin, xmax, 500)

    # calculate normal probability density function of sampling distribution
    # by using sigma/sqrt(n) as the sampling distribution variance
    ysamp = pdfnorm(x, sampDistMu, sampDistSigma)
    ylim = (0, ysamp.max())

    # plot sampling distribution 
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1, 1, 1)
    plots.scatter(x
        ,ysamp
        ,fig = fig
        ,ax = ax
        ,ylim = ylim
        ,title = ''
        ,markersize = 0
        ,linewidth = 2
        ,color = plots.RED)

    # calculate normal probability density function of population with population variance
    ypop = pdfnorm(x, mu, sigma)

    # get the largest y value for y axis limits
    ylim = ax.get_ylim()
    if ypop.max() > ylim[1]:
        ylim = (ylim[0], ylim[1])

    # plot the population distribution
    plots.scatter(x
        ,ypop
        ,fig = fig
        ,ax = ax
        ,ylim = ylim
        ,title = ''
        ,markersize = 0
        ,linewidth = 2
        ,color = plots.BLUE)

    # ----------------------------------------------------------------------
    # Plot areas that represent a 95% confidence interval. The population variance
    # sigma is assumed to be known. NOTE: The shaded region is represents a
    # range of x values that represent a 1 - alpha probability with respect
    # to the sampling distribution.
    # ----------------------------------------------------------------------
    alpha = 0.05

    # calculate the upper and lower z values that represent the standardized
    # upper and lower limits of the confidence interval
    zlo = stats.norm.ppf(alpha / 2)
    zhi = stats.norm.ppf(1 - (alpha / 2))

    # calculate the upper and lower x values that represent the
    # upper and lower limits of the confidence interval
    xlo = zlo * sigma / math.sqrt(n) + sampDistMu
    xhi = zhi * sigma / math.sqrt(n) + sampDistMu

    # create an array of x values that define the fill region
    xfill = np.linspace(xlo, xhi, 500)
    yfill = pdfnorm(xfill, sampDistMu, sampDistSigma)
    ax.fill_between(xfill, yfill, color = plots.RED)

    # plot the mean of the sampling distribution
    ax.plot(np.array([sampDistMu, sampDistMu])
        ,np.array((ylim))
        ,linewidth = 1.25
        ,color = plots.LIGHT_RED
        ,linestyle = 'dashed')

    # ----------------------------------------------------------------------
    # Build multiple confidence intervals by repeated sampling. Visualize.
    # NOTE: Not all confidence intervals contain the true mean mu.
    # ----------------------------------------------------------------------
    nDraw = 100     # number of samples to draw
    n = 20          # sample size

    # Create an array of y positions where the confidence intervals are to be drawn.
    # Exclude that first and last points so as not to plot on the edges of the plotting area.
    ypos = np.linspace(ylim[0], ylim[1], nDraw + 2)
    ypos = ypos[1:-1]

    # Visualization parameters
    whiskerWidth = 1e-3 / 2

    cntNoMean = 0
    for y in ypos:
        markersize = 3
        linewidth = 0.75
        color = plots.GREEN

        # draw a sample
        sample = np.random.choice(population, size = n)
        sampMean = sample.mean()

        # calculate the confidence interval
        cilo = zlo * sigma / math.sqrt(n) + sampMean
        cihi = zhi * sigma / math.sqrt(n) + sampMean

        # count confidence intervals that do not contain
        # the sampling distribution mean
        if sampDistMu < cilo or sampDistMu > cihi:
            cntNoMean = cntNoMean + 1
            markersize = 7
            linewidth = 2
            color = plots.ORANGE

        # visualize
        ax.plot(sampMean
            ,y
            ,marker = 'o'
            ,markersize = markersize
            ,linewidth = 0
            ,color = color)
        ax.plot([cilo, cihi]
            ,[y, y]
            ,linewidth = linewidth
            ,color = color)
        ax.plot([cilo, cilo]
            ,[y - whiskerWidth, y + whiskerWidth]
            ,linewidth = linewidth
            ,color = color)
        ax.plot([cihi, cihi]
            ,[y - whiskerWidth, y + whiskerWidth]
            ,linewidth = linewidth
            ,color = color)

    print('Out of {0} samples, {1} standard normal confidence intervals do not contain the sampling distribution mean.'\
        .format(nDraw, cntNoMean))
    print('')

    # format plot
    legend =\
    [
         mpl.lines.Line2D([0], [0], color = plots.RED, linewidth = 2, label = 'Sampling Distribution')
        ,mpl.lines.Line2D([0], [0], color = plots.BLUE, linewidth = 2, label = 'Population Distribution')
        ,mpl.lines.Line2D([0], [0], color = plots.LIGHT_RED, linewidth = 1.25, linestyle = 'dashed', label = 'Sampling Distribution Mean')
        ,mpl.patches.Patch(facecolor = plots.RED, label = '{0}% Probability Interval'.format(int((1 - alpha) * 100)))
        ,mpl.lines.Line2D([0], [0], color = plots.GREEN, linewidth = 1, label = '{0}% Confidence Interval'.format(int((1 - alpha) * 100)))
        ,mpl.lines.Line2D([0], [0], color = plots.ORANGE, linewidth = 2, label = 'Confidence Intervals that do not bound the sampling distribution mean')
    ]
    ax.legend(handles = legend)
    ax.set_xlabel('height')
    ax.set_ylabel('f(x)')
    ax.set_title('{0} Confidence Intervals using a Standard Normal Distribution'.format(nDraw))
    fig.tight_layout()

    # ----------------------------------------------------------------------
    # Student's t Distribution as Compared to the Standard Normal Distribution
    # ----------------------------------------------------------------------
    # Calculate probability density function values for the standard normal distribution
    yz = stats.norm.pdf(x, loc = mu, scale = sigma)

    # Draw a sample and caluculate the sample standard deviation.
    sample = np.random.choice(population, size = n)
    s = sample.std(ddof = 1)

    # Calculate the probability density function values for the t distribution.
    # df = n - 1 specifies to use n - 1 degrees of freedom
    yt = stats.t.pdf(x, df = n - 1, loc = mu, scale = s)

    # Visualize Student's t Distribution and compare to the standard normal distribution
    ylim = (0, max(yz.max(), yt.max()))
    fig, ax = plots.scatter(x, yz
        ,figsize = figsize
        ,ylim = ylim
        ,xlabel = 'height'
        ,ylabel = 'f(x)'
        ,linewidth = 2
        ,markersize = 0
        ,color = plots.BLUE)
    plots.scatter(x, yt
        ,fig = fig
        ,ax = ax
        ,ylim = ylim
        ,title = ''
        ,linewidth = 2
        ,markersize = 0
        ,color = plots.RED)

    # ----------------------------------------------------------------------
    # Build multiple confidence intervals using the t distribution.
    # ----------------------------------------------------------------------
    # calculate the upper and lower t values that represent the
    # upper and lower limits of the confidence interval using the t distribution
    tlo = stats.t.ppf(alpha / 2, df = n - 1)
    thi = stats.t.ppf(1 - (alpha / 2), df = n - 1)

    # upper and lower limits of the confidence interval
    xlo = tlo * sigma / math.sqrt(n) + sampDistMu
    xhi = thi * sigma / math.sqrt(n) + sampDistMu

    # create an array of x values that define the fill region
    xfill = np.linspace(xlo, xhi, 500)
    yfill = stats.t.pdf(xfill, n - 1, loc = sampDistMu, scale = s)
    ax.fill_between(xfill, yfill, color = plots.RED)

    # plot the mean of the sampling distribution
    ax.plot(np.array([sampDistMu, sampDistMu])
        ,np.array((ylim))
        ,linewidth = 1.25
        ,color = plots.LIGHT_RED
        ,linestyle = 'dashed')

    # Create an array of y positions where the confidence intervals are to be drawn.
    # Exclude that first and last points so as not to plot on the edges of the plotting area.
    ypos = np.linspace(ylim[0], ylim[1], nDraw + 2)
    ypos = ypos[1:-1]

    # Visualization parameters
    whiskerWidth = 1e-3 / 2

    cntNoMean = 0
    for y in ypos:
        markersize = 3
        linewidth = 0.75
        color = plots.GREEN

        # draw a sample
        sample = np.random.choice(population, size = n)
        sampMean = sample.mean()

        # calculate the confidence interval
        cilo = tlo * sigma / math.sqrt(n) + sampMean
        cihi = thi * sigma / math.sqrt(n) + sampMean

        # count confidence intervals that do not contain
        # the sampling distribution mean
        if sampDistMu < cilo or sampDistMu > cihi:
            cntNoMean = cntNoMean + 1
            markersize = 7
            linewidth = 2
            color = plots.ORANGE

        # visualize
        ax.plot(sampMean
            ,y
            ,marker = 'o'
            ,markersize = markersize
            ,linewidth = 0
            ,color = color)
        ax.plot([cilo, cihi]
            ,[y, y]
            ,linewidth = linewidth
            ,color = color)
        ax.plot([cilo, cilo]
            ,[y - whiskerWidth, y + whiskerWidth]
            ,linewidth = linewidth
            ,color = color)
        ax.plot([cihi, cihi]
            ,[y - whiskerWidth, y + whiskerWidth]
            ,linewidth = linewidth
            ,color = color)

    print('Out of {0} samples, {1} t distribution confidence intervals do not contain the sampling distribution mean.'\
        .format(nDraw, cntNoMean))
    print('')

    legend =\
    [
         mpl.lines.Line2D([0], [0], color = plots.BLUE, linewidth = 2, label = 'Standard Normal')
        ,mpl.lines.Line2D([0], [0], color = plots.RED, linewidth = 2, label = 't Distribution')
        ,mpl.lines.Line2D([0], [0], color = plots.LIGHT_RED, linewidth = 1.25, linestyle = 'dashed', label = 'Sampling Distribution Mean')
        ,mpl.patches.Patch(facecolor = plots.RED, label = '{0}% Probability Interval'.format(int((1 - alpha) * 100)))
        ,mpl.lines.Line2D([0], [0], color = plots.GREEN, linewidth = 1, label = '{0}% Confidence Interval'.format(int((1 - alpha) * 100)))
        ,mpl.lines.Line2D([0], [0], color = plots.ORANGE, linewidth = 2, label = 'Confidence Intervals that do not bound the sampling distribution mean')
    ]
    ax.legend(handles = legend)
    ax.set_title('{0} Confidence Intervals using a t Distribution'.format(nDraw))
    fig.tight_layout()





    plt.show()
