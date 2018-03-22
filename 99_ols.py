import subprocess as sp
import sklearn.datasets as ds
import pandas as pd
import sklearn.linear_model as lm
import numpy as np

sp.call( 'cls', shell = True )

#**********************************************************************
#
# DATA PREPARATION
#
#**********************************************************************
# load the diabetes dataset
diabetes = ds.load_diabetes()                                   # diabetes is an sklearn.utils.Bunch type
X = pd.DataFrame( diabetes.data )                               # features in a DataFrame
X.columns = diabetes.feature_names
Y = pd.Series( diabetes.target )                                # target as pandas.Series
Y.columns = ['price']
#print( 'Keys: \n' + str( diabetes.keys() ) )                    # keys of the diabetes dictionary
#print( 'Shape: \n' + str( diabetes.data.shape ) + '\n' )        # shape of dataset
#print( 'Features: \n' + str( diabetes.feature_names ) + '\n' )  # features
#print( 'Description: \n' + diabetes.DESCR + '\n' )              # description
#print( 'Head: \n' + str( X.head() ) )                           # first few rows of the dataset

#**********************************************************************
#
# MODEL FITTING
#
#**********************************************************************
lr = lm.LinearRegression()          # create LinearRegression object
lr.fit( X, Y )                      # fit a linear model

# create a DataFrame of coefficients and feature names
coef = pd.DataFrame( list( zip(X.columns, lr.coef_) ), columns = [ 'features', 'coefficients'] )

# append intercept to DataFrame
coef = coef.append( pd.DataFrame( list( zip( ['intercept']
                                           , [lr.intercept_] ) )
                                , columns = coef.columns ) )
print( 'Estimated intercept coefficient: ' + str( lr.intercept_ ) + '\n' )
print( 'Number of coefficients: ' + str( len( lr.coef_ ) ) + '\n' )
print( 'Coefficients:\n' + str( coef ) + '\n' )

# TODO: attempt to duplicate lr.predict using array multiplication
