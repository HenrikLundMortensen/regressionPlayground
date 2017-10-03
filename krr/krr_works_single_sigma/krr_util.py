import numpy as np


# def getAlpha(K,Y,gamma):
#     """
    
#     """
#     N = len(Y)
#     return np.linalg.inv(K+gamma*N*np.eye(K.shape[0]))@Y
    

def gaussKernel(x1,x2,sigma):
    """

    """
    return np.exp(-np.linalg.norm(x1-x2)**2/(2*sigma**2))




def predict(alpha,xtrain,x,sigma):
    """

    """
    return sum([alpha[i]*gaussKernel(xtrain[i],x,sigma) for i in range(len(alpha))])




    
