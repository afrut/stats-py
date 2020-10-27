#exec(open('descriptive.py').read())
import subprocess as sp
import scipy.stats as stats
import importlib as il
import matplotlib.pyplot as plt
import pickle as pk
import dfutl
import plots

if __name__ == '__main__':
    sp.call('cls', shell = True)
    il.reload(dfutl)
    il.reload(plots)

    # Load some data.
    with open('.\\iris.pkl','rb') as fl:
        df = pk.load(fl)
        cols = dfutl.numericColumns(df)
        df = df.loc[:, cols]

    # Numerical summaries of data
    print(df.describe())

    plots.stemleaf(df
        ,title = 'Stem and Leaf'
        ,save = True
        ,savepath = '.\\visual\\iris_stemleaf.txt')

    plots.histogram(df
        ,save = True
        ,savepath = '.\\visual\\iris_histogram.png'
        ,close = True)

    plots.boxplot(df
        ,save = True
        ,savepath = '.\\visual\\iris_boxplot.png'
        ,close = True)

    plots.scattermatrix(df
        ,save = True
        ,savepath = '.\\visual\\iris_scattermatrix.png'
        ,close = True)

    plots.heatmap(df
        ,save = True
        ,savepath = '.\\visual\\iris_heatmap.png'
        ,close = True)

    plots.probplot(df
        ,save = True
        ,savepath = '.\\visual\\iris_probplot.png'
        ,close = True)
