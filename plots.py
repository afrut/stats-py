import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as fgr
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import math

# ----------------------------------------
# boxplot
# ----------------------------------------
def boxplot(df
    ,fig = None
    ,figsize: tuple = (14.4, 9)
    ,title: str = None
    ,save: bool = False
    ,savepath: str = '.\\boxplot.png'
    ,show: bool = False
    ,close: bool = False):

    # ----------------------------------------------------------------------
    # process inputs
    # ----------------------------------------------------------------------
    if fig is None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(1,1,1)
    else:
        ax = fig.get_axes()[0]

    # ----------------------------------------------------------------------
    # actual plot
    # ----------------------------------------------------------------------
    ax = sns.boxplot(data = df, ax = ax)

    # ----------------------------------------------------------------------
    # formatting
    # ----------------------------------------------------------------------
    if title is not None:
        ax.set_title(title)

    if save:
        if savepath[-1] == '\\':
            savepath = savepath + 'boxplot.png'
        plt.savefig(savepath, format = 'png')

    if show:
        plt.show()

    if close:
        plt.close()

    return fig

# ----------------------------------------
# plot a histogram of certain variables
# ----------------------------------------
def histogram(df
    ,fig = None
    ,figsize: tuple = (14.4, 9)
    ,samefig: bool = True
    ,columns: list = None
    ,numBins: int = 10
    ,binWidth: float = None
    ,binStartVal: float = None
    ,barwidth: float = 0.35
    ,xlabel: str = None
    ,xlabels: list = None
    ,xlabelfontsize: int = 10
    ,xticklabelrotation: int = 30
    ,ylabel: str = None
    ,ylabels: list = None
    ,title: str = None
    ,tightLayout = False
    ,save: bool = False
    ,savepath: str = '.\\png\\histogram.png'
    ,show: bool = False
    ,close: bool = False):

    # ----------------------------------------------------------------------
    # process inputs
    # ----------------------------------------------------------------------
    if df is not None:
        if columns is not None:
            df = df.loc[:, columns]
        else:
            columns = df.columns
        numVar = len(columns)
    else:
        numVar = 0

    # ----------------------------------------------------------------------
    # infer data types of the input DataFrame
    # ----------------------------------------------------------------------
    isNumeric = np.vectorize(lambda x: np.issubdtype(x, np.number))
    colNumeric = isNumeric(df.dtypes)

    # if inputs are valid
    if numVar > 0:
        # determine the number of axes per row if all variables are
        # to be plot on same figure
        if(samefig):
            axPerRow = math.sqrt(numVar)
            ncols = int(math.ceil(axPerRow))
            numplots = 0
            nrows = 0
            while numplots < numVar:
                nrows = nrows + 1
                numplots = nrows * ncols

        # loop through all variables and plot them on the corresponding axes
        for cntAx in range(0, numVar):

            # get the series for which the histogram is to be made
            srs = df.iloc[:, cntAx]

            if colNumeric[cntAx]:
                # ----------------------------------------
                # infer the bins through the binWidth
                # ----------------------------------------
                if binWidth is not None:
                    # segregate data by the thickness of each bin
                    bins = list()
                    binStopVal = srs.max() + binWidth
                    if binStartVal is None:
                        binStartVal = srs.min()
                    bins = np.arange(binStartVal, binStopVal, binWidth)

                # ----------------------------------------
                # infer the bins through the number of bins
                # ----------------------------------------
                elif numBins is not None:
                    # segregate data by the number of bins
                    bins = np.linspace(srs.min(), srs.max(), numBins + 1)

                # ----------------------------------------
                # plot
                # ----------------------------------------
                # create the figure and plot
                if not samefig:
                    fig = plt.figure(figsize = figsize)
                    ax = fig.add_subplot(1,1,1)
                elif fig is None and samefig:
                    fig = plt.figure(figsize = figsize)
                    ax = fig.add_subplot(nrows,ncols, cntAx + 1)
                elif fig is not None and samefig:
                    ax = fig.add_subplot(nrows,ncols, cntAx + 1)
                lsVals, lsBins, _ = ax.hist(srs, bins = bins)

                # ----------------------------------------
                # format the plot
                # ----------------------------------------
                ax.set_xticks(lsBins)
                ax.set_xticklabels(np.round(lsBins, 4), rotation = xticklabelrotation)
                for ticklabel in ax.get_xticklabels():
                    ticklabel.set_fontsize(xlabelfontsize)
                ax.grid(linewidth = 0.5)
                ax.set_title(title)

                if xlabels is not None:
                    ax.set_xlabel(xlabels[cntAx])
                elif xlabel is not None:
                    ax.set_xlabel(xlabel)
                if ylabels is not None:
                    ax.set_ylabel(ylabels[cntAx])
                elif ylabel is not None:
                    ax.set_ylabel(ylabel)

                if title is not None:
                    ax.set_title(title)
                else:
                    ax.set_title(df.columns[cntAx])

            else:
                # ----------------------------------------
                # plot
                # ----------------------------------------
                # create the figure and plot
                if not samefig:
                    fig = plt.figure(figsize = figsize)
                    ax = fig.add_subplot(1,1,1)
                elif fig is None and samefig:
                    fig = plt.figure(figsize = figsize)
                    ax = fig.add_subplot(nrows,ncols, cntAx + 1)
                elif fig is not None and samefig:
                    ax = fig.add_subplot(nrows,ncols, cntAx + 1)
                x = np.array(list(set(srs)))
                y = df.iloc[:, [cntAx]].groupby(df.columns[cntAx]).size()
                barplot(x = x
                    ,y = y
                    ,fig = fig
                    ,ax = ax
                    ,grid = True
                    ,title = df.columns[cntAx]
                    ,tightLayout = True)

        if tightLayout:
            fig.tight_layout()

        if save:
            if savepath[-1:] == '\\':
                savepath = savepath + 'histogram.png'
            plt.savefig(savepath
                ,format = 'png')
        if show:
            plt.show()

        if close:
            plt.close()

                


# ----------------------------------------
# scatter matrix plot
# ----------------------------------------
def scattermatrix(
     dataframe
    ,columns: list = None
    ,figsize: tuple = (14.4, 9)
    ,save: bool = False
    ,savepath: str = '.\\scattermatrix.png'
    ,show: bool = False
    ,close: bool = False):

    if columns is None:
        columns = dataframe.columns.tolist()

    dfTemp = dataframe.loc[:, columns]
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1)
    axes = pd.plotting.scatter_matrix(dfTemp, ax = ax)
    for x in range(axes.shape[0]):
        for y in range(axes.shape[1]):
            ax = axes[x,y]
            ax.xaxis.label.set_rotation(30)
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.labelpad = 50

    if save:
        if savepath[-1:] == '\\':
            savepath = savepath + 'scattermatrix.png'
        plt.savefig(savepath, format = 'png')

    if show:
        plt.show()

    if close:
        plt.close()

# ----------------------------------------
# plot a heat map
# ----------------------------------------
def heatmap(df
    ,correlation = 0
    ,xcolumns: list = None
    ,ycolumns: list = None
    ,figsize: tuple = (14.4, 9)
    ,title: str = None
    ,save: bool = False
    ,savepath: str = '.\\heatmap.png'
    ,show: bool = False
    ,close: bool = False):

    # prepare variables for rows and columns
    if xcolumns is None:
        xcolumns = df.columns.tolist()
    if ycolumns is None:
        ycolumns = df.columns.tolist()

    # calculate correlations
    dfCorr = df.corr()
    dfCorr = dfCorr.loc[xcolumns, ycolumns]

    dfCorr = dfCorr.loc[xcolumns, ycolumns]
    dfZero = pd.DataFrame(np.zeros(shape = dfCorr.shape)
                         ,index = dfCorr.index
                         ,columns = dfCorr.columns)

    # bi-directionally mask correlations that are less than a certain threshold
    mask = dfCorr <= correlation
    mask = mask & (dfCorr >= correlation * -1)
    dfCorrMask = dfCorr.mask(mask, dfZero)

    # heat map of correlations
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1)
    ax = sns.heatmap(dfCorrMask, -1, 1, annot = True, annot_kws = dict([('fontsize', 6)]))
    ax.set_yticks([x + 0.5 for x in range(0, len(dfCorrMask.index))])
    ax.set_yticklabels(dfCorrMask.index)
    ax.set_xticks([x + 0.5 for x in range(0, len(dfCorrMask.columns))])
    ax.set_xticklabels(dfCorrMask.columns)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    if title is None:
        title = 'Correlation Threshold = {0:.3f}'.format(correlation)
    ax.set_title(title)

    if save:
        if savepath[-1:] == '\\':
            savepath = savepath + 'heatmap.png'
        plt.savefig(savepath
            ,format = 'png')
    if save:
        plt.savefig(savepath, format = 'png')

    if show:
        plt.show()

    if close:
        plt.close()

# ----------------------------------------
# simple scatter plot between two quantities
# ----------------------------------------
def scatter(x, y
    ,fig = None
    ,figsize: tuple = (14.4, 9)
    ,axesNew: bool = False
    ,xname: str = None
    ,yname: str = None
    ,xscale: str = None
    ,yscale: str = None
    ,marker: str = 'o'
    ,markersize: int = 5
    ,markeredgewidth: float = 0.4
    ,markeredgecolor: tuple = (0, 0, 0, 1)
    ,linewidth: int = 0
    ,color: tuple = (0, 0, 1, 1) # rgba
    ,title: str = None
    ,grid: bool = False
    ,save: bool = False
    ,savepath: str = '.\\scatterplot.png'
    ,show: bool = False
    ,close: bool = False):
    if fig is None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(1,1,1)
    elif axesNew:
        ax = fig.get_axes()[0].twinx()
    else:
        ax = fig.get_axes()[0]

    # check shape of data passed in
    oneX = len(x.shape) == 1 or min(x.shape) == 1
    multX = len(x.shape) == 2 and x.shape[1] > 1
    oneY =len(y.shape) == 1 or min(y.shape) == 1
    multY = len(y.shape) == 2 and y.shape[1] > 1

    if oneX:
        if oneY:
            lines = ax.plot(x, y
                ,marker = marker
                ,markersize = markersize
                ,linewidth = linewidth
                ,markeredgewidth = markeredgewidth
                ,markeredgecolor = markeredgecolor
                ,color = color)
        elif multY:
            numY = y.shape[1]
            colors = cm.brg(np.linspace(0, 1, numY))
            for cnt in range(0, numY):
                lines = ax.plot(x, y[:,cnt]
                    ,marker = marker
                    ,markersize = markersize
                    ,linewidth = linewidth
                    ,markeredgewidth = markeredgewidth
                    ,markeredgecolor = markeredgecolor
                    ,color = colors[cnt])
        else:
            print('Invalid input shapes x = {0}, y = {1}'.format(x.shape, y.shape))
    elif multX and multY:
        numX = x.shape[1]
        numY = y.shape[1]
        if numX == numY:
            colors = cm.brg(np.linspace(0, 1, numY))
            for cnt in range(0, numY):
                lines = ax.plot(x[:,cnt], y[:,cnt]
                    ,marker = marker
                    ,markersize = markersize
                    ,linewidth = linewidth
                    ,markeredgewidth = markeredgewidth
                    ,markeredgecolor = colors[cnt]
                    ,color = colors[cnt])
        else:
            print('Invalid input shapes x = {0}, y = {1}'.format(x.shape, y.shape))
    else:
        print('Invalid input shapes x = {0}, y = {1}'.format(x.shape, y.shape))
            

    if title is None:
        title = '{} vs {}'.format(yname, xname)

    if title is not None:
        ax.set_title(title)
    if xname is not None:
        ax.xaxis.label.set_text(xname)
    if yname is not None:
        ax.yaxis.label.set_text(yname)
    if grid:
        ax.grid(grid, linewidth = 0.5)

    if save:
        if savepath[-1:] == '\\':
            savepath = savepath + 'scatterplot.png'
        plt.savefig(savepath
            ,format = 'png')
    if show:
        plt.show()

    if close:
        plt.close()

    return fig

# ----------------------------------------
# plot a bar chart with text labels
# ----------------------------------------
def barplot(
     x = None
    ,y = None
    ,yLine = None
    ,fig = None
    ,figsize: tuple = (14.4, 9)
    ,ax = None
    ,yLim = None
    ,barwidth: float = 0.35
    ,xticklabels = None
    ,xlabel: str = None
    ,ylabel: str = None
    ,yLineLabel: str = None
    ,yLineLim = None
    ,grid = False
    ,title: str = None
    ,tightLayout = False
    ,save: bool = False
    ,savepath: str = ".\\barplot.png"
    ,show: bool = False
    ,close: bool = False):

    # Internal Function
    # Attach a text label above each bar in *rects*, displaying its height."""
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize = 7)

    # ----------------------------------------------------------------------
    # process inputs
    # ----------------------------------------------------------------------
    if y is None:
        y = np.arange(0,10)
        x = np.arange(len(y))
    elif x is None:
        x = np.arange(len(y))

    # ----------------------------------------------------------------------
    # actual plot
    # ----------------------------------------------------------------------
    if fig is None:
        fig = plt.figure(figsize = figsize)
    if ax is None:
        ax = fig.add_subplot(1,1,1)
    else:
        ax = ax
    rects = ax.bar(x, y)

    # plot a line if appropriate
    if yLine is not None:
        ax2 = ax.twinx()
        ax2.plot(x, yLine, color = "#ff6961")
        if yLineLabel is not None:
            ax2.set_ylabel(yLineLabel)
        if yLineLim is not None:
            ax2.set_ylim(yLineLim)

    # axis titles
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # set y limit
    if yLim is not None:
        ax.set_ylim(yLim)
    else:
        yMin = 0
        yMax = y.max()* 1.1
        ax.set_ylim([yMin, yMax])

    # grid lines
    if grid:
        ax.grid(True, linewidth = 0.5)

    # x-axis tick marks
    ax.set_xticks(x)

    # format xticklabels
    if xticklabels is None:
        xticklabels = x
    ax.set_xticklabels(xticklabels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # title and labels
    if title is not None:
        ax.set_title(title)

    autolabel(rects, ax)

    if tightLayout:
        fig.tight_layout()

    if save:
        if savepath[-1:] == '\\':
            savepath = savepath + 'barplot.png'
        plt.savefig(savepath
            ,format = 'png')
    if show:
        plt.show()

    if close:
        plt.close()

    return (fig, ax)

# ----------------------------------------
# plot a scatter plot between two quantities colored by a third
# ----------------------------------------
def colorscatter(x, y, z
    ,fig = None
    ,figsize: tuple = (14.4, 9)
    ,xname: str = 'x'
    ,yname: str = 'y'
    ,zname: str = 'z'
    ,numBins: int = 10
    ,title: str = None
    ,save: bool = False
    ,savepath: str = '.\\colorscatterplot.png'
    ,show: bool = False
    ,close: bool = False):

    if fig is None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(1,1,1)
    else:
        ax = fig.get_axes()[0]

    # create bins of z, 1 bin of z = 1 color shade of z
    valStart = z.min()
    valEnd = z.max()
    vals = np.linspace(valStart, valEnd, numBins + 1)
    colors = cm.seismic(np.linspace(0, 1, len(vals)))

    # loop through all bins
    for cnt in range(0, len(vals) - 1):

        # create a a boolean indexer for z values in current bin
        idx = (z >= vals[cnt]) & (z < vals[cnt + 1])

        # get all x and y values associated with this bin of z values
        xbin = x.loc[idx]
        ybin = y.loc[idx]

        # create a scatter plot
        fig = scatter(
             x = xbin
            ,y = ybin
            ,fig = fig
            ,xname = xname
            ,yname = yname
            ,markersize = 6
            ,color = tuple(colors[cnt])
            ,title = title
            ,save = save
            ,savepath = savepath
            ,show = show
            ,close = False)

    if title is not None:
        ax.set_title(title)
    if xname is not None:
        ax.xaxis.label.set_text(xname)
    if yname is not None:
        ax.yaxis.label.set_text(yname)

    if save:
        if savepath[-1:] == '\\':
            savepath = savepath + 'colorscatter.png'
        plt.savefig(savepath, format = 'png')

    if show:
        plt.show()

    if close:
        plt.close()

    return fig
