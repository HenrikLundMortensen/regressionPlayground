import numpy as np

def regGrad(X,y,w):
    """
    """
    n = len(y)
    return -2*X.T@(X@w-y)@w/n


def regGradVec(X,y,w):
    """
    """
    n = len(y)
    gradVec = np.array([(-2*X.T@(X@w-y))[i]*w[i]/n for i in range(len(w))])
    return gradVec



    

