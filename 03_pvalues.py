import subprocess as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sp.call( 'cls', shell=True )    # clear screen

# ----------------------------------------------------------------------
# 
# How to interpret p-values:
#
# Given an assumend population mean mu0, one-tailed p-values state the
# probability of encountering sample means greater than or equal to the
# one encountered.
#
# Given an assumed population mean mu0, two-tailed p-values state the
# probability of encountering sample means with the same or greater
# distance from the assumed population mean.
#
# Intuitively, given a distribution centered around mu with a standard
# deviation sigma, an xbar with a one-tailed p-value of 0.01 means
# the chance of encountering sample means that are "further" away
# from xbar in one direction is 1%. An xbar with a two-tailed p-value
# of 0.02 means the chance of encountering sample means that are
# "further" away from xbar in both directions is 2%.
#
# ----------------------------------------------------------------------

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

# draw a sample from the population with sample size of n
n = 50
sample = np.random.choice( population, size = n )
dfSample = pd.DataFrame( sample, columns = ['Individual Sample'] )
xbar = round( np.mean( sample ), 2 )
sx = round( np.std( sample ), 2 )
sexbar = round( sigma / (n**(1/2) ), 2 )
print( 'Sample mean = ' + str( round( xbar, 2 ) ) )
print( 'Sample standard deviation = ' + str( round( sx, 2 ) ) )
print( 'Standard error of sample mean = ' + str( sexbar ) )
print( '' )

# ----------------------------------------------------------------------
# 
# p-value calculations
#
# ----------------------------------------------------------------------
# calculate the z-score using the usually unknown population standard deviation
z = (xbar - mu) / sigma

# alternatively, calculate the z-score using the sample standard deviation
z = (xbar - mu) / sx

# find the one-tailed p-value
pvalue1 = 1 - stats.norm.cdf( z )

# find the two-tailed p-value
pvalue2 = ( 1 - stats.norm.cdf( z ) ) * 2

print( 'The one-tailed p-value = ' + str( pvalue1 ) )
print( 'The two-tailed p-value = ' + str( pvalue2 ) )
print( '' )
