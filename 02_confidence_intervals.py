import subprocess as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sp.call( 'cls', shell=True )    # clear screen

# ----------------------------------------------------------------------
# 
# If estimating the population mean is analogous to spear-fishing,
# confidence intervals are analogous to net fishing. They give a range
# of values that may contain the true population mean assuming a specific
# confidence level. They give an idea of how the sample mean varies
# around the true population mean.
#
# How to interpret confidence intervals:
#
# A 95% two-tailed interval around a sample mean means that if we take 100
# samples, calculate 100 sample means, and build 100 confidence
# intervals around the corresponding sample means, 5 of those confidence
# intervals will likely not contain the true population mean.
#
# One-tailed intervals follow the same logic.
#
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 
# Function to calculate two-tailed confidence intervals
#
# ----------------------------------------------------------------------
def twoTailCi( sample, clevel, sigma = None ):
    # calculate one-tailed confidence level
    clevel = clevel + (1 - clevel) / 2

    # calculate sample statistics
    xbar = round( np.mean( sample ), 4 )
    sx = round( np.std( sample ), 4 )

    # calculate z-score
    z = round( stats.norm.ppf( clevel ), 4 )

    # calculate an estimate of the standard error of the sample mean
    if sigma is None:
        sexbar = round( sx / ( np.sqrt(n) ), 4 )
    else:
        sexbar = round( sigma / ( np.sqrt(n) ), 4 )

    # calculate confidence interval bounds
    cihi = round( xbar + ( z * sexbar ), 4 )
    cilo = round( xbar - ( z * sexbar ), 4 )

    # recalculate two-tailed confidence level
    clevel = round( ( clevel - ( 1 - clevel ) ) * 100, 2 )

    print( str( clevel ) + '% confidence interval = ' +
           str( xbar ) + ' +/- ' + str(z) + ' * ' + str(sexbar) +
           ' = [' + str(cilo) + ', ' + str(cihi) + ']' )

# ----------------------------------------------------------------------
# 
# Function to calculate one-tailed confidence intervals
#
# ----------------------------------------------------------------------
def oneTailCi( sample, clevel, sigma = None ):
    # calculate sample statistics
    xbar = round( np.mean( sample ), 4 )
    sx = round( np.std( sample ), 4 )

    # calculate z-score
    z = round( stats.norm.ppf( clevel ), 4 )

    # calculate an estimate of the standard error of the sample mean
    if sigma is None:
        sexbar = round( sx / ( np.sqrt(n) ), 4 )
    else:
        sexbar = round( sigma / ( np.sqrt(n) ), 4 )

    # calculate confidence interval bounds
    cihi = round( xbar + ( z * sexbar ), 4 )
    cilo = round( xbar - ( z * sexbar ), 4 )

    clevel = round( clevel * 100 )
    print( str( clevel ) + '% confidence interval = ' +
           str( xbar ) + ' + ' + str(z) + ' * ' + str(sexbar) +
           ' = [-Inf, ' + str(cihi) + ']' )
    print( str( clevel ) + '% confidence interval = ' +
           str( xbar ) + ' - ' + str(z) + ' * ' + str(sexbar) +
           ' = [' + str(cilo) + ', +Inf]' )

# ----------------------------------------------------------------------
# 
# Main script
#
# ----------------------------------------------------------------------
# create a population of 100,000 numbers
mu = 160
sigma = 10
population = np.random.randn(10000) * sigma + mu
mu = np.mean(population)
sigma = np.std(population)
print( 'Population mean = ' + str( round( mu, 2 ) ) )
print( 'Population standard deviation = ' + str( round( sigma, 2 ) ) )
print( '' )

# histogram and Gaussian fit of population data
figWidth = 15
fig = plt.figure()
fig.set_size_inches( figWidth, figWidth / 1.6 )
ax = fig.add_subplot(1,2,1)
numBins = 100
freq, bins, patches = ax.hist( population, numBins )
binWidth = bins[1] - bins[0]
ax.set_xlabel( 'Bins' )
ax.set_ylabel( 'Frequency' )
ax.set_title( 'Histogram of Population with Gaussian Fit' )
x = np.linspace( bins[0], bins[-1], 100 )
y = stats.norm.pdf( x, mu, sigma ) * np.sum( freq * binWidth )
ax.plot( x, y )
ax.legend(['Gaussian Fit','Frequency'], loc='best')

# draw a sample from the population with sample size of n
n = 50
sample = np.random.choice( population, size = n )
xbar = round( np.mean( sample ), 2 )
sx = round( np.std( sample ), 2 )
sexbar = round( sigma / np.sqrt(n), 2 )
print( 'Sample mean = ' + str( round( xbar, 2 ) ) )
print( 'Sample standard deviation = ' + str( round( sx, 2 ) ) )
print( 'Standard error of sample mean = ' + str( sexbar ) )
print( '' )

# histogram and Gaussian fit of sample data
ax = fig.add_subplot(1,2,2)
numBins = 7
freq, bins, patches = ax.hist( sample, numBins )
binWidth = bins[1] - bins[0]
ax.set_xlabel( 'Bins' )
ax.set_ylabel( 'Frequency' )
ax.set_title( 'Histogram of Sample with Gaussian Fit' )
x = np.linspace( bins[0], bins[-1], 100 )
y = stats.norm.pdf( x, mu, sigma ) * np.sum( freq * binWidth )
ax.plot( x, y )
ax.legend(['Gaussian Fit','Frequency'], loc='best')

# ----------------------------------------------------------------------
# 
# Two-tailed confidence intervals
#
# ----------------------------------------------------------------------
print( 'Two-tailed confidence intervals using the population standard ' +
       'deviation to calculate sample mean standard error:' )
twoTailCi( sample, 0.8, sigma )
twoTailCi( sample, 0.9, sigma )
twoTailCi( sample, 0.95, sigma )
twoTailCi( sample, 0.99, sigma )
print( '' )

print( 'Two-tailed confidence intervals using the sample standard ' +
       'deviation to estimate sample mean standard error:' )
twoTailCi( sample, 0.8 )
twoTailCi( sample, 0.9 )
twoTailCi( sample, 0.95 )
twoTailCi( sample, 0.99 )
print( '' )

# ----------------------------------------------------------------------
# 
# One-tailed confidence intervals
#
# ----------------------------------------------------------------------
print( 'One-tailed confidence intervals using the poulation standard ' +
       'deviation to calculate sample mean standard error:' )
oneTailCi( sample, 0.8, sigma )
oneTailCi( sample, 0.9, sigma )
oneTailCi( sample, 0.95, sigma )
oneTailCi( sample, 0.99, sigma )
print( '' )

print( 'One-tailed confidence intervals using the sample standard ' +
       'deviation to estimate sample mean standard error:' )
oneTailCi( sample, 0.8 )
oneTailCi( sample, 0.9 )
oneTailCi( sample, 0.95 )
oneTailCi( sample, 0.99 )
print( '' )

# show all plots
plt.show()
