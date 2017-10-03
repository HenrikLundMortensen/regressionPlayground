import numpy as np
from matplotlib import pyplot as plt


def plotf(ax,f,xstart,xend):
    """
    """
    
    x = np.linspace(xstart,xend,100)
    y = f(x)

    p = ax.plot(x,y)
    return p


def plotPoints(ax,x,y):
    """
    """
    p = ax.plot(x,y,'o')
    return p


def plotPred(ax,KRR,xstart,xend):
    """
    """
    x = np.linspace(xstart,xend,400)
    y = KRR.predict(x)
    p = ax.plot(x,y)
    return p
