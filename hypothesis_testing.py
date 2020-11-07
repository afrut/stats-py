#exec(open('hypothesis_testing.py').read())
import subprocess as sp
import numpy as np
import pandas as pd
import importlib as il
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import plots
import confidence_intervals as ci

if __name__ == '__main__':
    sp.call('cls', shell = True)
    il.reload(plots)
    il.reload(ci)
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
    # Draw a random sample with which to perform hypothesis testing.
    # ----------------------------------------------------------------------
    n = 20
    samp = np.random.choice(population, size = n)
    sampmean = samp.mean()
    sampstd = samp.std(ddof = 1)
    print('Sample Characteristics:')
    print('Sample mean = ' + str(round(sampmean, 2)))
    print('Sample standard deviation = ' + str(round(sampstd, 2)))
    print('Sample range = [' +
           str(round(min(samp), 2)) + ',' +
           str(round(max(samp), 2)) + ']')
    print('')

    print('----------------------------------------------------------------------')
    print('  Two-sided Hypothesis Testing with Known Population Standard Deviation')
    print('----------------------------------------------------------------------')
    alpha = 0.05
    mu0 = 175
    xlo, xhi = ci.twoTail(alpha, n = n, sampmean = mu0, sigma = sigma)
    print('Sample mean = {0:.5}'.format(sampmean))
    print('Confidence Interval = {0:.5} <= x <= {1:.5}, alpha = {2:.2}'.format(xlo, xhi, alpha))
    if sampmean < xlo or sampmean > xhi:
        print('Reject H0: mu0 = {0} in favor of Ha: mu0 != {0}'.format(mu0))
    else:
        print('Fail to reject H0: mu0 = {0}'.format(mu0))
    print('')

    mu0 = 180
    xlo, xhi = ci.twoTail(alpha, n = n, sampmean = mu0, sigma = sigma)
    print('Sample mean = {0:.5}'.format(sampmean))
    print('Confidence Interval = {0:.5} <= x <= {1:.5}, alpha = {2:.2}'.format(xlo, xhi, alpha))
    if sampmean < xlo or sampmean > xhi:
        print('Reject H0: mu0 = {0} in favor of Ha: mu0 != {0}'.format(mu0))
    else:
        print('Fail to reject H0: mu0 = {0}'.format(mu0))
    print('')

    print('----------------------------------------------------------------------')
    print('  Two-sided Hypothesis Testing with Unknown Population Standard Deviation')
    print('----------------------------------------------------------------------')
    alpha = 0.05
    mu0 = 175
    xlo, xhi = ci.twoTail(alpha, n = n, sampmean = mu0, sampstd = sampstd)
    print('Sample mean = {0:.5}'.format(sampmean))
    print('Confidence Interval = {0:.5} <= x <= {1:.5}, alpha = {2:.2}'.format(xlo, xhi, alpha))
    if sampmean < xlo or sampmean > xhi:
        print('Reject H0: mu0 = {0} in favor of Ha: mu0 != {0}'.format(mu0))
    else:
        print('Fail to reject H0: mu0 = {0}'.format(mu0))
    print('')

    mu0 = 180
    xlo, xhi = ci.twoTail(alpha, n = n, sampmean = mu0, sampstd = sampstd)
    print('Sample mean = {0:.5}'.format(sampmean))
    print('Confidence Interval = {0:.5} <= x <= {1:.5}, alpha = {2:.2}'.format(xlo, xhi, alpha))
    if sampmean < xlo or sampmean > xhi:
        print('Reject H0: mu0 = {0} in favor of Ha: mu0 != {0}'.format(mu0))
    else:
        print('Fail to reject H0: mu0 = {0}'.format(mu0))
    print('')

    print('----------------------------------------------------------------------')
    print('  One-sided Lower-Bound Hypothesis Testing with Known Population Standard Deviation')
    print('----------------------------------------------------------------------')
    alpha = 0.05
    mu0 = 175
    xlo = ci.oneTailLo(alpha, n = n, sampmean = mu0, sigma = sigma)
    print('Sample mean = {0:.5}'.format(sampmean))
    print('Confidence Interval = x > {0:.5} , alpha = {1:.2}'.format(xlo, alpha))
    if sampmean < xlo:
        print('Reject H0: mu0 = {0} in favor of Ha: mu0 < {0}'.format(mu0))
    else:
        print('Fail to reject H0: mu0 = {0}'.format(mu0))
    print('')

    mu0 = 180
    xlo = ci.oneTailLo(alpha, n = n, sampmean = mu0, sigma = sigma)
    print('Sample mean = {0:.5}'.format(sampmean))
    print('Confidence Interval = x > {0:.5}, alpha = {1:.2}'.format(xlo, alpha))
    if sampmean < xlo:
        print('Reject H0: mu0 = {0} in favor of Ha: mu0 < {0}'.format(mu0))
    else:
        print('Fail to reject H0: mu0 = {0}'.format(mu0))
    print('')

    print('----------------------------------------------------------------------')
    print('  One-sided Lower-Bound Hypothesis Testing with Unknown Population Standard Deviation')
    print('----------------------------------------------------------------------')
    alpha = 0.05
    mu0 = 175
    xlo = ci.oneTailLo(alpha, n = n, sampmean = mu0, sampstd = sampstd)
    print('Sample mean = {0:.5}'.format(sampmean))
    print('Confidence Interval = x > {0:.5} , alpha = {1:.2}'.format(xlo, alpha))
    if sampmean < xlo:
        print('Reject H0: mu0 = {0} in favor of Ha: mu0 < {0}'.format(mu0))
    else:
        print('Fail to reject H0: mu0 = {0}'.format(mu0))
    print('')

    mu0 = 180
    xlo = ci.oneTailLo(alpha, n = n, sampmean = mu0, sampstd = sampstd)
    print('Sample mean = {0:.5}'.format(sampmean))
    print('Confidence Interval = x > {0:.5}, alpha = {1:.2}'.format(xlo, alpha))
    if sampmean < xlo:
        print('Reject H0: mu0 = {0} in favor of Ha: mu0 < {0}'.format(mu0))
    else:
        print('Fail to reject H0: mu0 = {0}'.format(mu0))
    print('')

    print('----------------------------------------------------------------------')
    print('  One-sided Upper-Bound Hypothesis Testing with Known Population Standard Deviation')
    print('----------------------------------------------------------------------')
    alpha = 0.05
    mu0 = 175
    xhi = ci.oneTailHi(alpha, n = n, sampmean = mu0, sigma = sigma)
    print('Sample mean = {0:.5}'.format(sampmean))
    print('Confidence Interval = x < {0:.5} , alpha = {1:.2}'.format(xhi, alpha))
    if sampmean > xhi:
        print('Reject H0: mu0 = {0} in favor of Ha: mu0 > {0}'.format(mu0))
    else:
        print('Fail to reject H0: mu0 = {0}'.format(mu0))
    print('')

    mu0 = 165
    xhi = ci.oneTailHi(alpha, n = n, sampmean = mu0, sigma = sigma)
    print('Sample mean = {0:.5}'.format(sampmean))
    print('Confidence Interval = x < {0:.5}, alpha = {1:.2}'.format(xhi, alpha))
    if sampmean > xhi:
        print('Reject H0: mu0 = {0} in favor of Ha: mu0 > {0}'.format(mu0))
    else:
        print('Fail to reject H0: mu0 = {0}'.format(mu0))
    print('')

    print('----------------------------------------------------------------------')
    print('  One-sided Upper-Bound Hypothesis Testing with Unknown Population Standard Deviation')
    print('----------------------------------------------------------------------')
    alpha = 0.05
    mu0 = 175
    xhi = ci.oneTailHi(alpha, n = n, sampmean = mu0, sampstd = sampstd)
    print('Sample mean = {0:.5}'.format(sampmean))
    print('Confidence Interval = x < {0:.5} , alpha = {1:.2}'.format(xhi, alpha))
    if sampmean > xhi:
        print('Reject H0: mu0 = {0} in favor of Ha: mu0 > {0}'.format(mu0))
    else:
        print('Fail to reject H0: mu0 = {0}'.format(mu0))
    print('')

    mu0 = 165
    xhi = ci.oneTailHi(alpha, n = n, sampmean = mu0, sampstd = sampstd)
    print('Sample mean = {0:.5}'.format(sampmean))
    print('Confidence Interval = x < {0:.5}, alpha = {1:.2}'.format(xhi, alpha))
    if sampmean > xhi:
        print('Reject H0: mu0 = {0} in favor of Ha: mu0 > {0}'.format(mu0))
    else:
        print('Fail to reject H0: mu0 = {0}'.format(mu0))
    print('')
