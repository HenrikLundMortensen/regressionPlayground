import numpy as np
from matplotlib import pyplot as plt
from generate_data import *
from controlCast import *
from scipy.optimize import minimize


inv = np.linalg.inv

def f(x):
    """
    Arguments:
    - `x`:
    """

    a,b,c,d,e = 1,1,5,10,1
    return a+b*x+c*x**2 + d*x**3 + e*np.sin(10*x)


def h(x,a):
    """
    Arguments:
    - `x`:
    """
    return sum([a[i]*x**i for i in range(len(a))])


def error(y,yhat):
    """
    """
    n = len(y)
    return 1/n*sum([(y[i]-yhat[i])**2 for i in range(n)])
    
def err_for_linesearch(gamma):
    """
    """
    reg = gamma*np.eye(n)
    atmp = inv(Xtrain.T@Xtrain + reg)@Xtrain.T@ytrain
    yhattest = h(xtest,atmp)
    return error(ytest,yhattest)

def linesearch(gamma,grad):
    """
    """
    res = minimize(lambda alpha: err_for_linesearch(gamma-alpha*grad),0.0001)
    return res.x
    


# stop
targetdeg = 3
Ntrain = 30
Ntest = 30
deg = 30
# deg = targetdeg


# generate_data(f,1,'train_data')
# generate_data(f,1,'test_data')
train_data = np.load('train_data.npy')
xtrain = train_data[0][500:500+Ntrain]
ytrain = train_data[1][500:500+Ntrain]

test_data = np.load('test_data.npy')
xtest = test_data[0][0:Ntest]
ytest = test_data[1][0:Ntest]


Xtrain = np.array(  [[x**i for i in range(deg+1)] for x in xtrain ]   )
Xtest = np.array(  [[x**i for i in range(deg+1)] for x in xtest ]   )




gamma = 0.01

minerr = 10
n = deg+1

for i in range(100):
    reg = gamma*np.eye(n)    
    a = inv(Xtrain.T@Xtrain + reg)@Xtrain.T@ytrain
    grad = regGrad(Xtest,ytest,a)
    alpha = linesearch(gamma,grad)
    gamma = gamma - alpha*grad
    yhattest = h(xtest,a)
    yhattrain = h(xtrain,a)
    # err = error(ytest,yhattest)
    # if err < minerr:
    #     minerr = err
    #     optgamma = gamma
    #     opta = a
    
    print('Gamma = %g, error_out = %g, error_in = %g, grad = %g, i = %g' %(gamma,error(ytest,yhattest),error(ytrain,yhattrain),grad,i))

stop

# gamma = 0.001
# reg = gamma*np.ones(deg+1)
# for i in range(100):
#     a = inv(Xtrain.T@Xtrain + reg)@Xtrain.T@ytrain
#     reg = reg - 0.00001*regGradVec(Xtest,ytest,a)
#     yhattest = h(xtest,a)
#     yhattrain = h(xtrain,a)
#     print('Error_out = %g, error_in = %g' %(error(ytest,yhattest),error(ytrain,yhattrain)))



# print(reg)


x = np.linspace(-1,1,100)





fig = plt.figure()
ax = fig.gca()
ax.plot(x,f(x),'black')
ax.plot(x,h(x,a),'red')
ax.plot(xtrain,ytrain,'bo')
ax.plot(xtest,ytest,'go')
# ax.plot(x[0:200],y[0:200],'ro')

ax.set_ylim([-8,20])

fig.savefig('regression.png')




gammalist = np.logspace(-10,1,100)
gradlist = []
errlist = []
for gamma in gammalist:
    reg = gamma*np.eye(n)    
    a = inv(Xtrain.T@Xtrain + reg)@Xtrain.T@ytrain
    grad = regGrad(Xtest,ytest,a)*0.01
    yhattest = h(xtest,a)
    err = error(ytest,yhattest)

    gradlist.append(grad)
    errlist.append(err*0.01)

numgrad = np.gradient(errlist)*n
    
fig = plt.figure()
ax = fig.gca()
ax.set_xscale('log')
ax.plot(gammalist,errlist)
ax.plot(gammalist,gradlist)
ax.plot(gammalist,numgrad)

fig.savefig('err_vs_gamma.png')
