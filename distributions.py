#exec(open('distributions.py').read())
# ----------------------------------------------------------------------
#
# Reference Material Montgomery & Runger: Applied statistics and Probability
# for Engineers 7ed
#
# ----------------------------------------------------------------------
import subprocess as sp
import importlib as il
import numpy as np
import scipy.stats as stats
import plots
import matplotlib as mpl
import matplotlib.pyplot as plt
import counting_techniques as ct
import math

if __name__ == '__main__':
    sp.call('cls', shell = True)
    il.reload(plots)
    plt.close()
    space = '    '

    # ----------------------------------------------------------------------
    # Plotting Parameters
    # ----------------------------------------------------------------------
    edgecolor = np.array([0.121568627,0.466666667,0.705882353]) / 1.6   # color of the edges of the bar graph rectangles
    show = False                                                        # whether or not to show plots

    # ----------------------------------------------------------------------
    #
    # Binomial Distribution
    # Consider the following situation: The probability of encountering a
    # poisonous molecule in a sample is 0.1. What is the probability of 5
    # samples containing a poisonous molecule in the next 20 samples? Take an
    # event that has a probability of occurrence p. The binomial distribution
    # gives the probability of this event occurring x times in the next n
    # trials.
    #
    # ----------------------------------------------------------------------
    print('----------------------------------------')
    print('  Binomial Distribution')
    print('----------------------------------------')
    # ----------------------------------------
    # probability mass function
    # ----------------------------------------
    p = 0.1     # the probability of an event E occurring
    x = 2       # the number of times E occurs
    n = 18      # in this many trials
    print('{0}An event E occurs with a probability of {1}.'.format(space, p))
    prob = stats.binom.pmf(x, n, p)
    prob2 = ct.comb(n, x) * p**x * (1 - p)**(n - x)
    assert(round(prob - prob2, 8) == 0)
    assert(0.284 - round(prob,3) == 0)
    print('{0}The probability that E occurs {1} times'.format(space, x) +\
        ' in the next {0} events is {1:.8}.'.format(n, prob))
    probs = list()
    xs = range(0, n + 1)
    for x in xs:
        prob = stats.binom.pmf(x, n, p)
        probs.append(round(prob, 8))
        print('{0}The probability that E occurs {1} times'.format(space, x) +\
            ' in the next {0} events is {1:.8}.'.format(n, prob))

    # ----------------------------------------
    # plotting
    # ----------------------------------------
    xs = np.array(xs)
    probs = np.array(probs)
    plots.barplot(xs, probs
        ,title = 'Binomial Distribution; p = {0:.8}, n = {1}, x = {2}'.format(p, n, x)
        ,align = 'edge'
        ,edgecolor = edgecolor
        ,show = False, close = False)
    print('')

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    x = 4
    prob = 0
    probcum = stats.binom.cdf(x - 1, n, p)
    probcum = 1 - probcum
    for x in range(0, x):
        prob = prob + stats.binom.pmf(x, n, p)
    prob = 1 - prob
    assert(round(abs(probcum - prob), 8) == 0)
    assert(abs(0.098 - round(probcum, 3)) == 0)
    print('{0}The probability that E occurs >= {1} times in {2} events {3:.8} is'\
        .format(space, x + 1, n, prob))

    x1 = 3
    x2 = 7
    prob = 0
    for x in range(x1, x2):
        prob = prob + stats.binom.pmf(x, n, p)
    probcum = stats.binom.cdf(x2, n, p) - stats.binom.cdf(x1, n, p)\
        - stats.binom.pmf(x2, n, p) + stats.binom.pmf(x1, n, p)
    assert(round(abs(prob - probcum), 8) == 0)
    print('{0}The probability that E occurs {1} <= x < {2} times in {3} events is {4:.8}'\
        .format(space, x1, x2, n, prob))

    # ----------------------------------------
    # mean and variance
    # ----------------------------------------
    mu = stats.binom.mean(n, p)
    sigmasq = stats.binom.var(n, p)
    assert(round(mu - (n * p), 8) == 0)
    assert(round(sigmasq - (n * p * (1 - p)), 8) == 0)
    print('{0}The mean of this binomial distribution is {1:.8}'.format(space, mu))
    print('{0}The variance of this binomial distribution is {1:.8}'.format(space, sigmasq))

    # ----------------------------------------------------------------------
    #
    # Geometric Distribution
    # Consider the following situation: The probability of encountering a
    # poisonous molecule in a sample is 0.1. What is the probability of
    # encountering the first sample with a poisonous molecule on the 10th
    # sample? How about the 15th? The geometric distribution answers these
    # questions. It gives the probability that the first occurrence of event E
    # (with probability p) occurs in exactly x events.
    #
    # ----------------------------------------------------------------------
    print('----------------------------------------')
    print('  Geometric Distribution')
    print('----------------------------------------')
    # ----------------------------------------
    # probability mass function
    # ----------------------------------------
    p = 0.1     # probability of an event E occuring
    x = 5       # the number of trials performed before E occurs
    prob = stats.geom.pmf(x, p)
    prob2 = (1 - p)**(x - 1) * p
    assert(round(prob - prob2, 8) == 0)
    assert(0.066 == round(prob, 3))
    print('{0}An event E occurs with a probability of {1}.'.format(space, p))
    print('{0}The probability that E first occurs in {1} times is {2:.8}'\
        .format(space, x, prob))
    probs = list()
    xs = range(x + 1)
    for x in xs:
        prob = stats.geom.pmf(x, p)
        probs.append(round(prob, 8))
        print('{0}The probability that E first occurs in {1} times is {2:.8}'\
            .format(space, x, prob))

    # ----------------------------------------
    # plotting
    # ----------------------------------------
    xs = np.array(xs)
    probs = np.array(probs)
    plots.barplot(xs, probs
        ,title = 'Geometric Distribution; p = {0:.8}, n = {1}, x = {2}'.format(p, n, x)
        ,align = 'edge'
        ,edgecolor = edgecolor
        ,show = False, close = False)
    print('')

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    x = 4
    prob = 0
    for x in range(0, x):
        prob = prob + stats.geom.pmf(x, p)
    prob = 1 - prob
    assert(0.729 == round(prob, 3))
    print('{0}The probability that E first occurs in >= {1} events is {2:.8}'.format(space, x + 1, prob))

    x1 = 3
    x2 = 7
    prob = 0
    for x in range(x1, x2):
        prob = prob + stats.geom.pmf(x, p)
    probcum = stats.geom.cdf(x2, p) - stats.geom.cdf(x1, p)\
        - stats.geom.pmf(x2, p) + stats.geom.pmf(x1, p)
    assert(round(abs(probcum - prob), 8) == 0)
    print('{0}The probability that first E occurs in {1} <= x < {2} events is {3:.8}'.format(space, x1, x2, prob))

    # ----------------------------------------
    # mean and variance
    # ----------------------------------------
    mu = stats.geom.mean(p)
    sigmasq = stats.geom.var(p)
    assert(round(mu - (1 / p), 8) == 0)
    assert(round(sigmasq - ((1 - p) / p**2), 8) == 0)
    print('{0}The mean of this geometric distribution is {1:.8}'.format(space, mu))
    print('{0}The variance of this geometric distribution is {1:.8}'.format(space, sigmasq))
    print('')

    # ----------------------------------------------------------------------
    #
    # Negative Binomial Distribution
    # This is similar to the Geometric Distribution with a slight twist.
    #
    # Consider the following situation: The probability of encountering a
    # poisonous molecule in a sample is 0.1. What is the probability of
    # encountering the exactly 3 samples with a poisonous molecule by the 10th
    # sample? How about exactly 5 samples by the 15th? The Negative Binomial
    # Distribution answers these questions. It gives the probability that the
    # rth occurrence of event E (with probability p) occurs by exactly x events.
    #
    # ----------------------------------------------------------------------
    print('----------------------------------------')
    print('  Negative Binomial Distribution')
    print('----------------------------------------')
    # ----------------------------------------
    # probability mass function
    # ----------------------------------------
    p = 0.1
    x = 10          # number of events
    r = 4           # number of occurrences of event E
    pplot = p
    xplot = x
    rplot = r
    print('{0}An event E occurs with a probability of {1}.'.format(space, p))
    # pmf(num trials - num event E, num event E, probability)
    prob = stats.nbinom.pmf(x - r, r, p)
    print('{0}The probability that E occurs exactly {1} times by the {2}th event is {3:.8}'\
        .format(space, r, x, prob))
    assert(round(abs(prob - 0.004464),6) == 0)
    xs = range(4, x + 1)
    probs = list()
    for x in xs:
        prob = stats.nbinom.pmf(x - r, r, p)
        probs.append(round(prob, 8))
        print('{0}The probability that E occurs exactly {1} times by the {2}th event is {3:.8}'\
            .format(space, r, x, prob))

    # ----------------------------------------
    # plotting
    # ----------------------------------------
    xs = np.array(xs)
    probs = np.array(probs)
    plots.barplot(xs, probs
        ,title = 'Negative Binomial Distribution; p = {0:.8}, n = {1}, r = {2}'\
            .format(pplot, xplot, rplot)
        ,align = 'edge'
        ,edgecolor = edgecolor
        ,show = False, close = False)

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    x = 5
    r = 3
    p  = 0.2
    print('')
    print('{0}An event E occurs with a probability of {1}.'.format(space, p))
    prob = 0
    xs = range(r, x + 1)
    for x in xs:
        prob = prob + stats.nbinom.pmf(x - r, r, p)
    probcum = stats.nbinom.cdf(x - r, r, p)
    assert(round(abs(prob - probcum), 8) == 0)
    assert(0.058 == round(probcum,3))
    print('{0}The probability that E occurs exactly {1} times by the <= {2}th event is {3:.8}'\
        .format(space, r, x, prob))

    # ----------------------------------------
    # mean and variance
    # ----------------------------------------
    mu = stats.nbinom.mean(r, p, r)
    sigmasq = stats.nbinom.var(r, p)
    assert(round(mu - (r / p), 8) == 0)
    assert(round(sigmasq - (r * (1 - p) / p**2), 8) == 0)
    print('{0}The mean of this negative binomial distribution is {1:.8}'.format(space, mu))
    print('{0}The variance of this negative binomial distribution is {1:.8}'.format(space, sigmasq))
    print('')

    # ----------------------------------------------------------------------
    #
    # Hypergeometric Distribution
    # Consider the following situation: 850 parts produced contains 50
    # defective parts. Two parts are selected at random without replacement.
    # What is the probability that both parts are defective? What is the
    # probability that both parts are not defective? What is the probability
    # that only 1 of the parts is defective?
    #
    # Suppose that the event of interest is a part being defective. The
    # Hypergeometric Distribution gives the probability of the sample
    # containing x = 2 defective parts. In general, consider a pool of N = 850
    # objects, K = 50 of which are of interest. A sample of size n = 2 is drawn
    # from this pool. The Hypergeometric Distribution gives the probability
    # that x = 1 of these objects is of interest.
    #
    # ----------------------------------------------------------------------
    print('----------------------------------------')
    print('  Hypergeometric Distribution')
    print('----------------------------------------')
    # ----------------------------------------
    # probability mass function
    # ----------------------------------------
    N = 850     # number of total objects
    K = 50      # number of objects of interest
    n = 2       # number of objects drawn from pool as a sample
    x = 1       # number of objects of interest in the sample
    prob = stats.hypergeom.pmf(x, N, K, n)
    prob2 = ct.comb(K, x) * ct.comb(N - K, n - x) / ct.comb(N, n)
    assert(round(prob - prob2, 8) == 0)
    assert(0.111 - round(prob,3) == 0)
    print('{0}A pool contains N = {1} objects.'.format(space, N))
    print('{0}K = {1} of these objects are of interest.'.format(space, K))
    print('{0}A sample of size n = {1} is drawn from the pool.'.format(space, n))
    print('{0}The probability that the pool contains x = {1} objects of interest is {2:.8}.'\
        .format(space, x, prob))
    probs = list()
    xs = range(0, n + 1)
    for x in xs:
        prob = stats.hypergeom.pmf(x, N, K, n)
        probs.append(round(prob, 8))
        print('{0}The probability that the sample contains x = {1} objects of interest is {2:.8}.'\
            .format(space, x, prob))

    # ----------------------------------------
    # plotting
    # ----------------------------------------
    xs = np.array(xs)
    probs = np.array(probs)
    plots.barplot(xs, probs
        ,title = 'Hypergeometric Distribution; N = {0}, K = {1}, n = {2}'.format(N, K, n)
        ,align = 'edge'
        ,edgecolor = edgecolor
        ,show = False, close = False)
    print('')

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    N = 300
    K = 100
    n = 4
    x = 2
    probcum = stats.hypergeom.cdf(x, N, K, n)
    probcum = 1 - probcum + stats.hypergeom.pmf(x, N, K, n)
    prob = 0
    xs = range(0, x)
    for xl in xs:
        prob = prob + stats.hypergeom.pmf(xl, N, K, n)
    prob = 1 - prob
    assert(round(probcum - prob, 8) == 0)
    assert(0.407 - round(probcum, 3) == 0)
    print('{0}The probability that the sample of size n = {1} contains'.format(space, n) +
        ' x >= {0} objects of interest is {1:.8}'.format(x, probcum))

    # ----------------------------------------
    # mean and variance
    # ----------------------------------------
    mu = stats.hypergeom.mean(N, K, n)
    sigmasq = stats.hypergeom.var(N, K, n)
    p = K / N
    assert(round(mu - (n * p), 8) == 0)
    assert(round(sigmasq - (n * p * (1 - p) * ((N - n)/(N - 1))), 8) == 0)
    print('{0}The mean of this hypergeometric distribution is {1:.8}'.format(space, mu))
    print('{0}The variance of this hypergeometric distribution is {1:.8}'.format(space, sigmasq))
    print('')

    # ----------------------------------------------------------------------
    #
    # Poisson Distribution
    # Consider the following situation: Flaws occur at random along the length
    # of a thin copper wire. It is given that the average number of flaws per
    # mm = lambda. What is the probability of encounter x = 10 flaws
    # in T = 5 mm of wire? The Poisson Distribution answers these kinds of
    # questions.
    #
    # In general, given an average number of occurrences of event E per unit
    # parameter(lambda), the Poisson Distribution gives the probability of
    # event E occurring x times in T units.
    #
    # ----------------------------------------------------------------------
    print('----------------------------------------')
    print('  Poisson Distribution')
    print('----------------------------------------')
    # ----------------------------------------
    # probability mass function
    # ----------------------------------------
    lamb = 2.3      # number of times event E occurs per unit on average
    x = 10          # number of times event E occurs
    T = 5           # unit examined
    prob = stats.poisson.pmf(x, lamb * T)
    prob2 = math.exp(-lamb * T) * (lamb * T)**x / math.factorial(x)
    assert(round(prob - prob2, 8) == 0)
    assert(0.113 - round(prob, 3) == 0)
    print('{0}An event E occurs {1} times per unit on average'.format(space, lamb))
    print('{0}The number of units examined T = {1}'.format(space, T))
    print('{0}The probability of E occurring x = {1} times is {2:.8}'\
        .format(space, x, prob))
    probs = list()
    xs = range(0, x + 5)
    for xl in xs:
        prob = stats.poisson.pmf(xl, lamb * T)
        probs.append(prob)
        print('{0}The probability that E occurs {1} times'.format(space, xl) +\
            ' in T = {0} units is {1:.8}'.format(T, prob))
    
    # ----------------------------------------
    # plotting
    # ----------------------------------------
    xs = np.array(xs)
    probs = np.array(probs)
    plots.barplot(xs, probs
        ,title = 'Poisson Distribution; lamb = {0}, T = {1}'.format(lamb, T)
        ,align = 'edge'
        ,edgecolor = edgecolor
        ,show = False, close = False)
    print('')

    # ----------------------------------------
    # cumulative probabilities
    # ----------------------------------------
    x = 1
    T = 2
    probcum = stats.poisson.cdf(x, lamb * T)
    probcum = 1 - probcum + stats.poisson.pmf(x, lamb * T)
    prob = 0
    for xl in range(0, x):
        prob = prob + stats.poisson.pmf(xl, lamb * T)
    prob = 1 - prob
    assert(round(prob - probcum, 8) == 0)
    assert(0.9899 - round(probcum, 4) == 0)
    print('{0}The probability that E occurs x >= {1} times P(x >= {2}) = {3:.8}'\
        .format(space, x, x, probcum))

    # ----------------------------------------
    # mean and variance
    # ----------------------------------------
    mu = stats.poisson.mean(lamb * T)
    sigmasq = stats.poisson.var(lamb * T)
    assert(round(mu - (lamb * T), 8) == 0)
    assert(round(sigmasq - (lamb * T), 8) == 0)
    print('{0}The mean of this poisson distribution is {1:.8}'.format(space, mu))
    print('{0}The variance of this poisson distribution is {1:.8}'.format(space, sigmasq))


    # show plots
    if(show):
        plt.show()
    else:
        plt.close()
