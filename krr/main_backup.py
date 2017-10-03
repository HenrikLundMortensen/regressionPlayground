import numpy as np
from krr_util import *
from krr import *
from matplotlib import pyplot as plt


def f(x):
    """

    """

    return np.cos(0.5*np.pi*x) + 1


x = np.linspace(1,20,100)
y = f(x)

noise = 0.05

Ntrain = 40
xtrain = np.linspace(1,20,Ntrain)
ytrain = f(xtrain) + noise*np.random.normal(size=Ntrain)

Nval = 13
xval = np.linspace(1,20,Nval)
yval = f(xval)+ noise*np.random.normal(size=Nval)

sigma = 1.1
gamma = 0.0001

KRR = regressor(kernel=gaussKernel,
                xtrain=xtrain,
                ytrain=ytrain,
                xval=xval,
                yval=yval,
                sigma=sigma,
                gamma=gamma)

KRR.getAlpha()
print(KRR.dLds())
print('Init err = %g' %(KRR.getError(xval,yval)))

beta = 0.1
grad = 1
i = 0
M = 100
Hclist = np.zeros(M)
while i < M:
    grad = KRR.dLds()
    KRR.sigma = KRR.sigma - beta*grad
    print('sigma = %2.2f, grad = %2.4f, error = %2.4f' %(KRR.sigma,grad,KRR.getError(xval,yval)))
    i += 1
    bestsigma = KRR.sigma
# bestsigma = KRR.sigma

i = 0
gradlist = np.zeros(M)
sigmalist = np.linspace(0.2,2,M)
errorlist = np.linspace(0.2,2,M)

# for sigma in np.flipud(sigmalist):
#     KRR.sigma = sigma
#     gradlist[i] = KRR.dLds()
#     errorlist[i] = KRR.getError(xval,yval)

#     i += 1

# # predList = KRR.predict(x)
KRR.sigma = bestsigma
KRR.getAlpha()
fig = plt.figure()
ax = fig.gca()
ax.plot(x,KRR.predict(x),linewidth=5)
ax.plot(xtrain,ytrain,'ro')
ax.plot(xval,yval,'ko')
ax.plot(x,f(x))
fig.savefig('krr.png')

fig = plt.figure()
ax = fig.gca()
# ax.plot(x,KRR.predict(x),linewidth=5)
ax.plot(xtrain,ytrain,'ro')
j = 0
for xt in xtrain:
    if np.mod(j,3) ==0:
        NKernelList = 100
        kernelList = np.zeros(NKernelList)
        xlocal = np.linspace(xt-3,xt+3,NKernelList)
        i = 0
        for xl in xlocal:
            kernelList[i] = KRR.alpha[j]*gaussKernel(xt,xl,KRR.sigma)
            i+=1
        ax.plot(xlocal,kernelList,'green',linewidth=1)
        j+=1
    else:
        j+=1

# ax.plot(xval,yval,'ko')
ax.plot(x,f(x))
fig.savefig('kernels.png')

# fig = plt.figure()
# ax = fig.gca()
# ax.plot(sigmalist,Hclist)
# # ax.set_xlim([0.5,1.5])
# # ax.set_ylim([min(Hclist),max(Hclist)])
# fig.show()


# fig = plt.figure()
# ax = fig.gca()
# ax.plot(sigmalist,gradlist)
# # ax.plot(sigmalist,np.gradient(errorlist))
# # ax.set_xlim([0.5,1.5])
# # ax.set_ylim([-3,3])
# fig.show()

# fig = plt.figure()
# ax = fig.gca()
# ax.plot(sigmalist,np.log(errorlist))
# # ax.set_xlim([0.5,1.5])
# # ax.set_ylim([-0.005,0.005])
# fig.show()


