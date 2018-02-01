import subprocess as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sp.call( 'cls', shell=True )    # clear screen

# ----------------------------------------------------------------------
# 
# Population ditribution
# 
# ----------------------------------------------------------------------
# create a population of 100,000
mu = 160
sigma = 10
population = np.random.randn(10000) * sigma + mu;
mu = np.mean( population )                # population mean
sigma = np.std( population, ddof = 0 )    # population standard deviation
print( 'Population mean = ' + str( round( mu, 2 ) ) )
print( 'Population standard deviation = ' + str( round( sigma, 2 ) ) )
print( 'Population range = [' +
       str( round( min( population ), 2 ) ) + ',' +
       str( round( max( population ), 2 ) ) + ']' )
print( '' )
dfPopulation = pd.DataFrame( population, columns = ['Population Distribution'] )

# ----------------------------------------------------------------------
# 
# Single point estimate from a sample
# 
# ----------------------------------------------------------------------
#
# The sample mean is our best estimate for the population mean.
# 
# ----------------------------------------------------------------------
# get a sample from from the population with sample size of n
n = 50
sample = np.random.choice( population, size = n )
dfSample = pd.DataFrame( sample, columns = ['Individual Sample'] )
xbar = round( np.mean( sample ), 2 )        # sample mean
sx = round( np.std( sample, ddof = 0 ), 2 ) # sample standard deviation
print( 'Sample mean = ' + str( round( xbar, 2 ) ) )
print( 'Sample standard deviation = ' + str( round( sx, 2 ) ) )
print( '' )

# ----------------------------------------------------------------------
# 
# Sampling distribution
# 
# ----------------------------------------------------------------------
#
# Sample means can vary between samples. If we take 5 samples and
# calculate the sample means, they will all be slightly different. The
# natural question is, how do sample means vary with regards to the
# population mean? It turns out that sample means can be thought of as
# belonging to a sampling distribution whose mean is the population mean and
# whose standard deviation = sigma / (sqrt(n)). The standard deviation
# of the sampling distribution is called the standard error of the
# sample mean. It gives us an idea of how sample means vary around
# the true population mean. Thus, the sample mean can be thought of
# as a value taken from the sampling distribution.
#
# Disclaimer: The population standard deviation is usually not known
# when calculating the standard error of the sample mean. This is known
# here because this is a a simulation.
# However, when the sampling size, n, is greater than or equal to 30
# the sampling distribution is approximately normal and the standard
# deviation of the sample can be used in place of the population
# standard deviation.
#
# ----------------------------------------------------------------------
# create the sampling distribution
lsXBar = list()
for cnt in range( 0, 100000 ):
    sample = np.random.choice( population, size = n )
    lsXBar.append( np.mean( sample ) )
dfSamplingDist = pd.DataFrame( lsXBar, columns = ['Sampling Distribution'] )
xbars = np.array( lsXBar )
sexbar = round( np.std( xbars ), 2 )
sexbarCalc = round( sigma / ( np.sqrt( n ) ), 2 )
sexbarEst = round( stats.sem( sample, ddof = 0 ), 4 )
# calculate statistics on the sampling distribution
print( 'Sampling distribution mean = ' + str( round( np.mean( xbars ), 2 ) ) )
print( 'Sampling distribution standard deviation = ' + str( sexbar ) )
print( 'Sampling distribution standard deviation as ' +
       'standard error of sample mean = ' + str( sexbar ) )
print( 'Sampling distribution standard deviation calculated from ' +
       'population standard deviation = ' + str( sexbarCalc ) )
print( 'Estimate of standard error of sample mean = ' +
       str( sexbarEst ) )
print( '' )

# ----------------------------------------------------------------------
# 
# Plotting
# 
# ----------------------------------------------------------------------
numBins = 40

dfPopulation.hist( bins = numBins )     # histogram of population
dfSample.hist( bins = numBins )         # histogram of sample
dfSamplingDist.hist( bins = numBins )   # histogram of sampling distribution

# normal probability plot of population distribution
fig = plt.figure()
ax = fig.add_subplot(111)
stats.probplot( population, plot = plt )
ax.set_title( 'Normal Probability Plot of the Population' )

# normal probability plot of the sample
fig = plt.figure()
ax = fig.add_subplot(111)
stats.probplot( sample, plot = plt )
ax.set_title( 'Normal Probability Plot of the Sample' )

# normal probability plot of sample means
fig = plt.figure()
ax = fig.add_subplot(111)
stats.probplot( xbars, plot = plt )
ax.set_title( 'Normal Probability Plot of Sample Means' )

# show all plots
plt.show()
