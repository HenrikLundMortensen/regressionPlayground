import numpy as np
from krr_util import *
from krr import *
from matplotlib import pyplot as plt
from plots import *
from scipy.optimize import minimize


def f(x):
    """
    """
    # np.cos(0.5*np.pi*x) + 1 +
    res = np.cos(0.5*np.pi*x) 
    for i in range(len(x)):
        if x[i] >= 4:
            res[i] += + 0.2*np.sin(8*np.pi*x[i])
        
    return res


def linesearch(KRR,grad):
    """
    """
    
    res = minimize(lambda alpha: err_for_linesearch(gamma-alpha*grad),0.0001)

    return res.x

Ntrain = 35
Nval = 30

noise = 0.0

xtrain = np.linspace(0,5,Ntrain)
xtrain = np.append(np.linspace(0,4,10),np.linspace(4.1,5,25))
# print(xtrain)
ytrain = f(xtrain)+noise*np.random.normal(size=Ntrain)

xval = np.linspace(0.2,4.8,Nval)
xval = np.append(np.linspace(0.2,4.8,15),np.linspace(4.1,4.8,15))
yval = f(xval)+noise*np.random.normal(size=Nval)


sigma = 0.7
sigmaVec = np.array([sigma for i in range(Ntrain)])
sigmaVec = np.append(np.linspace(1,1,10),np.linspace(0.05,0.05,25))
gamma = 0.00001

KRR = regressor(kernel=gaussKernel,
                xtrain=xtrain,
                ytrain=ytrain,
                xval=xval,
                yval=yval,
                sigma=sigma,
                sigmaVec = sigmaVec,
                gamma=gamma)
KRR.getAlpha()

beta = 0.01
M = 200
tol = 2e-5

# for i in range(M):
#     gradVec = KRR.dLdsi()
#     KRR.sigmaVec = KRR.sigmaVec - beta * gradVec
#     KRR.getAlpha()
#     err = KRR.getError(xval,yval)
#     print('error = %2.10f, i = %i/%i, |grad| = %1.10f' %(err,i,M,np.linalg.norm(gradVec)))
#     if np.linalg.norm(gradVec) < tol:
#         print('Breaking')
#         break
    # # print(KRR.sigmaVec)    
    

bestsigmaVec = KRR.sigmaVec
minerr = 1

# beta = 10
# M = 500
# tol = 1e-5

# for i in range(M):
#     grad = KRR.dLds()
#     KRR.sigma = KRR.sigma - beta * grad
#     KRR.getAlpha()
#     err = KRR.getError(xval,yval)
#     if abs(grad) < tol:
#         break
#     print('sigma = %2.2f, grad = %2.4f, error = %2.4f' %(KRR.sigma,grad,err))

# bestsigma = KRR.sigma
# minerr = err




# sigmalist = np.linspace(0.2,2,1000)
# errorList = np.zeros(len(sigmalist))
# gradList = np.zeros(len(sigmalist))
# i = 0
# for s in sigmalist:
#     KRR.sigma = s
#     KRR.getAlpha()
#     gradList[i] = KRR.dLds()
#     errorList[i] = KRR.getError(xval,yval)
#     i += 1



    
KRR.getAlpha()
fig = plt.figure()
ax = fig.gca()
plotf(ax,f,0,5)
plotPoints(ax,xtrain,ytrain)
plotPoints(ax,xval,yval)
plotPred(ax,KRR,0,5)
fig.savefig('functionPlot.png')


# fig = plt.figure()
# ax = fig.gca()
# ax.plot(sigmalist,np.log(errorList))
# ax.plot(bestsigma,np.log(minerr),'ro')
# fig.savefig('errorPlot.png')

# fig = plt.figure()
# ax = fig.gca()
# ax.plot(sigmalist,gradList)
# ax.plot(bestsigma,0,'ro')
# fig.savefig('gradPlot.png')




fig = plt.figure()
ax = fig.gca()
ax.plot(xtrain,np.log(KRR.sigmaVec))
# ax.set_ylim([-2,6])
fig.savefig('sigmaVec.png')




fig = plt.figure()
ax = fig.gca()

j = 0
for xt in xtrain:
    if np.mod(j,1) ==0:
        NKernelList = 100
        kernelList = np.zeros(NKernelList)
        xlocal = np.linspace(xt-5,xt+5,NKernelList)
        i = 0
        for xl in xlocal:
            kernelList[i] = KRR.alpha[j]*gaussKernel(xt,xl,KRR.sigmaVec[j])
            # KRR.alpha[j]*gaussKernel(xt,xl,KRR.sigmaVec[j])
            i+=1
        ax.plot(xlocal,kernelList,'green',linewidth=1)
        j+=1
    else:
        j+=1
        
ax.set_xlim([min(xtrain),max(xtrain)])
fig.savefig('kernels.png')





