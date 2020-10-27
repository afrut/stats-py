#exec(open('.\\distributions\\bivariate.py').read())
import subprocess as sp
import pandas as pd
import pickle as pk

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # Load a dataset of the response times vs the number of bars as signal strength.
    # The index is the response time in seconds. The columns are values of x. The
    # data is the probability of observing the (x,y) value pair. For example, the probability
    # of having a response time of 2 seconds with 3 bars of signal strength is 0.2.
    with open('.\\distributions\\responseTime.pkl','rb') as fl:
        df = pk.load(fl)

    # This data is effectively a joint probability distribution in tabular form.
    x = 3
    y = 2
    print('The probability of x = {0} and y = {1} is {2}.'\
        .format(x, y, df.loc[y, '{0}'.format(3)]))
    print('')

    # The marginal probability distribution of X and Y can be obtained from their
    # joint probability distribution by "summing out" either variable.
    mpdX = df.sum(axis = 0)
    mpdY = df.sum(axis = 1)
    print('The marginal probability distribution of X is:')
    print(mpdX)
    print('')
    print('The marginal probability distribution of Y is:')
    print(mpdY)
    print('')

    # These marginal probability distributions can be used to determine the
    # expectation of each variable by multiplying the value of the variable
    # by their probabilities and summing.
    xvals = mpdX.index.values.astype(float)
    probs = mpdX.values
    Ex = (xvals * probs).sum()
    yvals = mpdY.index.values.astype(float)
    probs = mpdY.values
    Ey = (yvals * probs).sum()
    print('The expectations of X and Y are:')
    print('E(X) = {0:.6}'.format(Ex))
    print('E(Y) = {0:.6}'.format(Ey))
