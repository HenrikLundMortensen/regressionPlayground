import numpy as np
from krr_util import *
from krr import *
from matplotlib import pyplot as plt
from plots import *


def f(x):
    """
    """

    


    
    return np.cos(0.5*np.pi*x) + 1


Ntrain = 10
Nval = 10

noise = 0.0

xtrain = np.linspace(0,5,Ntrain)
# xtrain = np.append(np.linspace(0,3,5),np.linspace(3.1,5,10))
# print(xtrain)
ytrain = f(xtrain)+noise*np.random.normal(size=Ntrain)

xval = np.linspace(0.2,4.8,Nval)
yval = f(xval)+noise*np.random.normal(size=Nval)


sigma = 1.5
gamma = 0.001

KRR = regressor(kernel=gaussKernel,
                xtrain=xtrain,
                ytrain=ytrain,
                xval=xval,
                yval=yval,
                sigma=sigma,
                gamma=gamma)
KRR.getAlpha()











beta = 10
M = 500
tol = 1e-5

for i in range(M):
    grad = KRR.dLds()
    KRR.sigma = KRR.sigma - beta * grad
    KRR.getAlpha()
    err = KRR.getError(xval,yval)
    if abs(grad) < tol:
        break
    print('sigma = %2.2f, grad = %2.4f, error = %2.4f' %(KRR.sigma,grad,err))

bestsigma = KRR.sigma
minerr = err












sigmalist = np.linspace(0.2,2,1000)
errorList = np.zeros(len(sigmalist))
gradList = np.zeros(len(sigmalist))
i = 0
for s in sigmalist:
    KRR.sigma = s
    KRR.getAlpha()
    gradList[i] = KRR.dLds()
    errorList[i] = KRR.getError(xval,yval)
    i += 1











    
KRR.sigma = bestsigma
KRR.getAlpha()
fig = plt.figure()
ax = fig.gca()
plotf(ax,f,0,5)
plotPoints(ax,xtrain,ytrain)
plotPoints(ax,xval,yval)
plotPred(ax,KRR,0,5)
fig.savefig('functionPlot.png')


fig = plt.figure()
ax = fig.gca()
ax.plot(sigmalist,np.log(errorList))
ax.plot(bestsigma,np.log(minerr),'ro')
fig.savefig('errorPlot.png')

fig = plt.figure()
ax = fig.gca()
ax.plot(sigmalist,gradList)
ax.plot(bestsigma,0,'ro')
fig.savefig('gradPlot.png')
