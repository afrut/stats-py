import subprocess as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sp.call( 'cls', shell=True )    # clear screen

# ----------------------------------------------------------------------
# 
# Main script
#
# ----------------------------------------------------------------------
# create a population of 100,000 numbers
mu = [ 160,175 ]
sigma = [ 10,15 ]
population = [ np.random.randn(10000) * sigma[0] + mu[0]
             , np.random.randn(10000) * sigma[1] + mu[1] ]
mu = [ round( np.mean(population[0]) )
     , round( np.mean(population[1]) ) ]
sigma = [ round( np.std(population[0]), 4 )
        , round( np.std(population[1]), 4 ) ]
print( 'Population 0 mean = ' + str( round( mu[0], 2 ) ) )
print( 'Population 0 standard deviation = ' + str( round( sigma[0], 2 ) ) )
print( 'Population 1 mean = ' + str( round( mu[1], 2 ) ) )
print( 'Population 1 standard deviation = ' + str( round( sigma[1], 2 ) ) )
print( '' )

# draw a sample from the population with sample size of n
n = [ 23,67 ]
sample = list()
sample.append( np.random.choice( population[0], size = n[0] ) )
sample.append( np.random.choice( population[1], size = n[1] ) )
xbar = [ round( np.mean( sample[0] ), 2 )
       , round( np.mean( sample[1] ), 2 ) ]
sx = [ round( np.std( sample[0] ), 2 )
     , round( np.std( sample[1] ), 2 ) ]
sexbar = [ round( sigma[0] / ( n[0]**(1/2) ), 2 )
         , round( sigma[1] / ( n[1]**(1/2) ), 2 ) ]
print( str( stats.sem( sample[0] ) ) )
print( 'Sample 0 mean = ' + str( round( xbar[0], 2 ) ) )
print( 'Sample 0 standard deviation = ' + str( round( sx[0], 2 ) ) )
print( 'Standard error of sample 1 mean = ' + str( sexbar[0] ) )
print( 'Sample 1 mean = ' + str( round( xbar[1], 2 ) ) )
print( 'Sample 1 standard deviation = ' + str( round( sx[1], 2 ) ) )
print( 'Standard error of sample 2 mean = ' + str( sexbar[1] ) )
print( '' )

# ----------------------------------------------------------------------
# 
# Hypothesis test for inequality
#
# ----------------------------------------------------------------------
print( 'Assertion: Population 0 mean is not equal to population 1 mean' )
print( 'H0: mu[0] = ' + str( mu[1] ) )
print( 'HA: mu[0] != ' + str( mu[1] ) ) 
