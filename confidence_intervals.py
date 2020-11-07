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

# ----------------------------------------------------------------------
# Function to calculate the probability density function values of the normal
# distribution.
# ----------------------------------------------------------------------
def pdfnorm(x, mu, sigma):
    return 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((x - mu)**2) / (2 * sigma**2))

# ----------------------------------------------------------------------
# Two-tail confidence intervals
# ----------------------------------------------------------------------
def twoTail(alpha, n = None, sampmean = None, sigma = None, sampstd = None, samp = None):
    ret = None

    if n == None and samp is not None:
        n = len(samp)

    # check if a sample mean has been passed in
    if sampmean is None and samp is not None:
        sampmean = samp.mean()

    # check if the population standard deviation is unknown
    if sigma == None:

        # t distribution always needs the sample size for degrees of freedom
        assert(n is not None)

        # use t distribution
        tlo = stats.t.ppf(alpha / 2, df = n - 1)
        thi = stats.t.ppf(1 - (alpha / 2), df = n - 1)

        # check if sample standard deviation is provided or can be calculated
        if sampstd == None and samp is not None:
            sampstd = samp.std(ddof = 1)

        # check if x values or t values are to be returned
        if sampmean is not None and sampstd is not None:
            xlo = sampmean + (tlo * (sampstd / math.sqrt(n)))
            xhi = sampmean + (thi * (sampstd / math.sqrt(n)))
            ret = (xlo, xhi)
        else:
            ret = (thi, tlo)

    else:

        # use standard normal distribution
        zlo = stats.norm.ppf(alpha / 2)
        zhi = stats.norm.ppf(1 - (alpha / 2))

        # check if x values or z values are to be returned
        if sampmean is not None and n is not None:
            xlo = sampmean + (zlo * (sigma / math.sqrt(n)))
            xhi = sampmean + (zhi * (sigma / math.sqrt(n)))
            ret = (xlo, xhi)
        else:
            ret = (zlo, zhi)

    return ret

# ----------------------------------------------------------------------
# One-tail lower-bound confidence intervals
# ----------------------------------------------------------------------
def oneTailLo(alpha, n = None, sampmean = None, sigma = None, sampstd = None, samp = None):
    ret = None

    if n == None and samp is not None:
        n = len(samp)

    # check if a sample mean has been passed in
    if sampmean is None and samp is not None:
        sampmean = samp.mean()

    # check if the population standard deviation is unknown
    if sigma == None:

        # t distribution always needs the sample size for degrees of freedom
        assert(n is not None)

        # use t distribution
        tlo = stats.t.ppf(alpha, df = n - 1)

        # check if sample standard deviation is provided or can be calculated
        if sampstd == None and samp is not None:
            sampstd = samp.std(ddof = 1)

        # check if x values or t values are to be returned
        if sampmean is not None and sampstd is not None:
            xlo = sampmean + (tlo * (sampstd / math.sqrt(n)))
            ret = xlo
        else:
            ret = tlo

    else:

        # use standard normal distribution
        zlo = stats.norm.ppf(alpha)

        # check if x values or z values are to be returned
        if sampmean is not None and n is not None:
            xlo = sampmean + (zlo * (sigma / math.sqrt(n)))
            ret = xlo
        else:
            ret = zlo

    return ret

# ----------------------------------------------------------------------
# One-tail upper-bound confidence intervals
# ----------------------------------------------------------------------
def oneTailHi(alpha, n = None, sampmean = None, sigma = None, sampstd = None, samp = None):
    ret = None

    if n == None and samp is not None:
        n = len(samp)

    # check if a sample mean has been passed in
    if sampmean is None and samp is not None:
        sampmean = samp.mean()

    # check if the population standard deviation is unknown
    if sigma == None:

        # t distribution always needs the sample size for degrees of freedom
        assert(n is not None)

        # use t distribution
        thi = stats.t.ppf(1 - alpha, df = n - 1)

        # check if sample standard deviation is provided or can be calculated
        if sampstd == None and samp is not None:
            sampstd = samp.std(ddof = 1)

        # check if x values or t values are to be returned
        if sampmean is not None and sampstd is not None:
            xhi = sampmean + (thi * (sampstd / math.sqrt(n)))
            ret = xhi
        else:
            ret = thi

    else:

        # use standard normal distribution
        zhi = stats.norm.ppf(1 - alpha)

        # check if x values or z values are to be returned
        if sampmean is not None and n is not None:
            xhi = sampmean + (zhi * (sigma / math.sqrt(n)))
            ret = xhi
        else:
            ret = zhi

    return ret

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
    # Calculate the sampling distribution's properties given that we know
    # the standard deviation of the population
    # ----------------------------------------------------------------------
    sampDistMu = mu
    sampDistSigma = sigma / math.sqrt(n)

    # ----------------------------------------------------------------------
    # Visualize one-sided vs two-sided confidence intervals.
    # ----------------------------------------------------------------------
    n = 20
    figsize = (14.4, 9)

    # define the minimum and maximum x values for the sampling distribution
    # to be 3 times the sampling distribution's standard deviation
    xmin = mu - 3 * (sigma / math.sqrt(n))
    xmax = mu + 3 * (sigma / math.sqrt(n))

    # create an array of x-values over which to calculate the pdf values of the
    # sampling distribution
    x = np.linspace(xmin, xmax, 500)

    # define the significance level
    alpha = 0.05

    # calculate values of the probability function
    y = stats.norm.pdf(x, loc = mu, scale = sigma / math.sqrt(n))

    # calculate the high and low values of x corresponding to a two-tailed
    # confidence interval
    xlotest = stats.norm.ppf(alpha / 2, loc = mu, scale = sampDistSigma)
    xhitest = stats.norm.ppf(1 - (alpha / 2), loc = mu, scale = sampDistSigma)
    xlo, xhi = twoTail(alpha, n = n, sampmean = mu, sigma = sigma)
    assert(abs(xlo - xlotest) < 1e-8)
    assert(abs(xhi - xhitest) < 1e-8)

    # initialize plotting parameters
    nplot = 1
    nrow = 2
    ncol = 2
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(nrow, ncol, nplot)

    # plot the population distribution
    plots.scatter(x, y
        ,fig = fig
        ,ax = ax
        ,title = 'Two-sided Confidence Interval'
        ,xlabel = 'height'
        ,ylabel = 'f(x)'
        ,linewidth = 2
        ,markersize = 0)

    # fill the areas corresponding to the significance level of a
    # 2-sided confidence interval
    xfill = x[x <= xlo]
    yfill = y[x <= xlo]
    ax.fill_between(xfill, yfill, color = plots.BLUE)
    xfill = x[x >= xhi]
    yfill = y[x >= xhi]
    ax.fill_between(xfill, yfill, color = plots.BLUE)
    nplot = nplot + 1

    # fill areas corresponding to the significance level of an
    # lower bound confidence interval
    ax = fig.add_subplot(nrow, ncol, nplot)
    plots.scatter(x, y
        ,fig = fig
        ,ax = ax
        ,title = 'Lower Bound Confidence Interval'
        ,markersize = 0
        ,linewidth = 2
        ,xlabel = 'height'
        ,ylabel = 'f(x)'
        ,color = plots.BLUE)
    xlotest = stats.norm.ppf(alpha, loc = mu, scale = sampDistSigma)
    xlo = oneTailLo(alpha, n = n, sampmean = mu, sigma = sigma)
    assert(abs(xlo - xlotest) < 1e-8)
    xfill = x[x <= xlo]
    yfill = y[x <= xlo]
    ax.fill_between(xfill, yfill, color = plots.BLUE)
    nplot = nplot + 1

    # fill areas corresponding to the significance level of an
    # upper bound confidence interval
    ax = fig.add_subplot(nrow, ncol, nplot)
    plots.scatter(x, y
        ,fig = fig
        ,ax = ax
        ,title = 'Upper Bound Confidence Interval'
        ,markersize = 0
        ,linewidth = 2
        ,xlabel = 'height'
        ,ylabel = 'f(x)'
        ,color = plots.BLUE)
    xhitest = stats.norm.ppf(1 - alpha, loc = mu, scale = sampDistSigma)
    xhi = oneTailHi(alpha, n = n, sampmean = mu, sigma = sigma)
    xfill = x[x >= xhi]
    yfill = y[x >= xhi]
    ax.fill_between(xfill, yfill, color = plots.BLUE)
    nplot = nplot + 1
    fig.suptitle('Sampling Distrubitions with alpha = {0:.2}'.format(alpha))
    fig.tight_layout()

    # ----------------------------------------------------------------------
    # Plot the corrensponding normal distributions of the population and sampling
    # distributions. NOTE: The sampling distribution is much "tighter" implying
    # a smaller variance. Note that both are centered around the same value.
    # ----------------------------------------------------------------------
    x = dfPop.values
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
    # calculate the upper and lower x values that represent the
    # upper and lower limits of the confidence interval
    xlo, xhi = twoTail(alpha, n = n, sampmean = sampDistMu, sigma = sigma)

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
        cilo, cihi = twoTail(alpha, n = n, sampmean = sampMean, sigma = sigma)

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
    yt = stats.t.pdf(x, df = n - 1, loc = mu, scale = s / math.sqrt(n))

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
    xlo, xhi = twoTail(alpha, n = n, sampmean = sampDistMu, sampstd = s)

    # create an array of x values that define the fill region
    xfill = np.linspace(xlo, xhi, 500)
    yfill = stats.t.pdf(xfill, n - 1, loc = sampDistMu, scale = s / math.sqrt(n))
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
        s = sample.std(ddof = 1)

        # calculate the confidence interval
        cilo, cihi = twoTail(alpha, n = n, sampmean = sampMean, sampstd = s)

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
         mpl.lines.Line2D([0], [0], color = plots.BLUE, linewidth = 2, label = 'Standard Normal Population Distribution')
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
