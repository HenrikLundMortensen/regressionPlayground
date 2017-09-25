import numpy as np



def generate_data(f,noise_fac,name):
    """
    """
    N = 1000
    ind = np.arange(N)
    ind = np.random.permutation(ind)

    x = np.linspace(-0.8,0.8,N)
    y = f(x) + np.random.normal(size=N)*noise_fac

    x = x[ind]
    y = y[ind]


    np.save(name,np.array([x,y]))
