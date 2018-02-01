import subprocess as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sp.call( 'cls', shell=True )    # clear screen

# ----------------------------------------------------------------------
# 
# When calculating confidence intervals and p-values, a z-score is
# needed. In calcualting the z-score, one needs the population
# standard deviation sigma. However, this is not known in most cases so
# an appropriate substitute would be standard deviation of the
# sample sx. Substituting sx for sigma becomes less of an issue
# when sample sizes are greater than 30.
#
# However, it is more appropriate to use the t-distribution and t-scores
# as opposed to the standard normal distribution and z-scores when
# calculating confidence intervals and p-values with the sample standard
# deviation sx.
#
# The t-distribution is how a statistic t = (xbar - mu0)/sx varies
# around 0. To define a t-distribution, one needs to specify the degrees
# of freedom df, which is equal to sample size - 1.
#
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 
# Function to calculate two-tailed confidence intervals
#
# ----------------------------------------------------------------------
def twoTailCi( sample, clevel ):
    # calculate one-tailed confidence level
    clevel = clevel + (1 - clevel) / 2

    # calculate sample statistics
    xbar = round( np.mean( sample ), 4 )
    sx = round( np.std( sample ), 4 )
    n = len( sample )
    df = n - 1

    # calculate t-score
    t = round( stats.t.ppf( clevel, df ), 4 )

    # calculate standard error of sample mean
    sexbar = round( sx / ( np.sqrt(n) ), 4 )

    # calculate confidence interval bounds
    cihi = round( xbar + ( t * sexbar ), 4 )
    cilo = round( xbar - ( t * sexbar ), 4 )

    # recalculate two-tailed confidence level
    clevel = round( ( clevel - ( 1 - clevel ) ) * 100, 2 )

    print( str( clevel ) + '% confidence interval = ' +
           str( xbar ) + ' +/- ' + str(t) + ' * ' + str(sexbar) +
           ' = [' + str(cilo) + ', ' + str(cihi) + ']' )

# ----------------------------------------------------------------------
# 
# Function to calculate one-tailed confidence intervals
#
# ----------------------------------------------------------------------
def oneTailCi( sample, clevel ):
    # calculate sample statistics
    xbar = round( np.mean( sample ), 4 )
    sx = round( np.std( sample ), 4 )
    n = len( sample )
    df = n - 1

    # calculate z-score
    t = round( stats.norm.ppf( clevel ), 4 )

    # calculate an estimate of the standard error of the sample mean
    sexbar = round( sx / ( np.sqrt(n) ) ), 4 )

    # calculate confidence interval bounds
    cihi = round( xbar + ( t * sexbar ), 4 )
    cilo = round( xbar - ( t * sexbar ), 4 )

    clevel = round( clevel * 100, 2)
    print( str( clevel ) + '% confidence interval = ' +
           str( xbar ) + ' + ' + str(t) + ' * ' + str(sexbar) +
           ' = [-Inf, ' + str(cihi) + ']' )
    print( str( clevel ) + '% confidence interval = ' +
           str( xbar ) + ' - ' + str(t) + ' * ' + str(sexbar) +
           ' = [' + str(cilo) + ', +Inf]' )

# ----------------------------------------------------------------------
# 
# Function to calculate one-tailed p-values
#
# ----------------------------------------------------------------------
def oneTailPvalue( sample, mu0 ):
    # calculate sample statistics
    xbar = np.mean( sample )
    sx = np.std( sample )
    n = len( sample )
    df = n - 1

    # calculate the standard error of the ma
    sexbar = round( sx / ( np.sqrt(n) ), 4 )

    # calculae t-score
    t = round( abs( ( xbar - mu0 ) ) / sexbar, 4 )

    # calculate the pvalue
    pvalue = round( 1 - stats.t.cdf( t, df ), 4 )

    print( 'One-tailed p-value = ' + str( pvalue ) + ' with mu0 = ' +
           str( mu0 ) )

# ----------------------------------------------------------------------
# 
# Function to calculate two-tailed p-values
#
# ----------------------------------------------------------------------
def twoTailPvalue( sample, mu0 ):
    # calculate sample statistics
    xbar = np.mean( sample )
    sx = np.std( sample )
    n = len( sample )
    df = n - 1

    # calculate the standard error of the ma
    sexbar = round( sx / ( np.sqrt(n) ), 4 )

    # calculae t-score
    t = round( abs( ( xbar - mu0 ) ) / sexbar, 4 )

    # calculate the pvalue
    pvalue = round( ( 1 - stats.t.cdf( t, df ) ) * 2, 4 )

    print( 'Two-tailed p-value = ' + str( pvalue ) + ' with mu0 = ' +
           str( mu0 ) )

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
n = 23
sample = np.random.choice( population, size = n )
xbar = round( np.mean( sample ), 2 )
sx = round( np.std( sample ), 2 )
sexbar = round( sx / (n**(1/2) ), 2 )
print( 'Sample mean = ' + str( round( xbar, 2 ) ) )
print( 'Sample standard deviation = ' + str( round( sx, 2 ) ) )
print( 'Standard error of sample mean = ' + str( sexbar ) )
print( '' )

# ----------------------------------------------------------------------
# 
# Two-tailed confidence intervals
#
# ----------------------------------------------------------------------
print( 'Two-tailed confidence intervals using the t-distribution: ' )
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
print( 'One-tailed confidence intervals using the t-distribution: ' )
oneTailCi( sample, 0.8 )
oneTailCi( sample, 0.9 )
oneTailCi( sample, 0.95 )
oneTailCi( sample, 0.99 )
print( '' )

# ----------------------------------------------------------------------
# 
# One-tailed p-value
#
# ----------------------------------------------------------------------
print( 'One-tailed p-values using the t-distribution: ' )
oneTailPvalue( sample, 155 )
oneTailPvalue( sample, 157.5 )
oneTailPvalue( sample, 160 )
oneTailPvalue( sample, 162.5 )
oneTailPvalue( sample, 165 )
print( '' )

print( 'Two-tailed p-values using the t-distribution: ' )
twoTailPvalue( sample, 155 )
twoTailPvalue( sample, 157.5 )
twoTailPvalue( sample, 160 )
twoTailPvalue( sample, 162.5 )
twoTailPvalue( sample, 165 )
