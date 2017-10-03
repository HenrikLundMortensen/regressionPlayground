from krr_util import *

class regressor():

    def __init__(self,xtrain,ytrain,kernel,sigma=1,sigmaVec=None,gamma=1,xval=None,yval=None):
        """

        """
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.N = len(ytrain)
        self.kernel = kernel
        self.sigma = sigma
        self.sigmaVec = sigmaVec
        self.gamma = gamma
        self.xval = xval
        self.yval = yval
        

    # def getK(self):
    #     """
    #     """
    #     xt = self.xtrain
    #     k=self.kernel
    #     s = self.sigma
    #     return = np.array([[k(xt[i],xt[j],s) for i in range(self.N)] for j in range(self.N)])

    # def getKgrad(self):
    #     """

    #     """
    #     s = self.sigma
    #     xt = self.xtrain
    #     k = self.kernel
    #     return np.array([[1/(s**3)* (np.linalg.norm(xt[i]-xt[j])**2)*k(xt[i],xt[j],s)
    #                             for i in range(self.N)]
    #                            for j in range(self.N)])

    def getK(self,xt,x):
        """
        """
        
        k = self.kernel
        sVec = self.sigmaVec
        K = np.array([[k(xt[i],x[j],sVec[i]) for i in range(len(xt))] for j in range(len(x))])
        # print('kernel(%g,%g) =%g, sigma = %g' %(xt[0],x[5],k(xt[0],x[5],s),s))
        return K
        
    def getKgrad(self,xt,x):
        """
        """
        
        k = self.kernel
        s = self.sigma
        
        return np.array([[1/(s**3)* (np.linalg.norm(xt[i]-x[j])**2)*k(xt[i],x[j],s)
                          for i in range(len(xt))]
                         for j in range(len(x))])


    def getAlpha(self):
        """
        """
        
        xt = self.xtrain
        self.Kt = self.getK(xt,xt)
        g = self.gamma
        yt = self.ytrain
        N = self.N

        self.alpha = np.linalg.inv(self.Kt+N*g*np.eye(self.Kt.shape[0]))@yt

    def predict(self,x):
        """
        """
        
        xt = self.xtrain
        K = self.getK(xt,x)
        a = self.alpha
        return K@a


    def dLds(self):
        """
        """

        s = self.sigma
        g = self.gamma
        xval = self.xval
        xtrain = self.xtrain
        Yval = self.yval
        Ytrain = self.ytrain
        Ntrain = self.N
        Nval = len(Yval)
        
        self.getAlpha()
        a = self.alpha
        Kval = self.getK(xtrain,xval)
        Ktrain = self.getK(xtrain,xtrain)
        dKvalds = self.getKgrad(xtrain,xval)
        dKtrainds = self.getKgrad(xtrain,xtrain)

        
        p = -2/Nval * Kval.T@(Yval-Kval@a)@np.linalg.inv(2/Ntrain *Ktrain + g*np.eye(Ktrain.shape[0]))
        dEvalds  = -1/(Nval)*(  a.T@dKvalds.T@(Yval - Kval@a) + (Yval-Kval@a).T@dKvalds@a)

        return dEvalds - p@(2/Ntrain*dKtrainds@a)



    def dKds(self,xi,xj,si):
        """
        """
        k = self.kernel
        return 1/(si**3)* (np.linalg.norm(xi-xj)**2)*k(xi,xj,si)
        
    

    def dLdsi(self):
        """
        """
        s = self.sigma
        sVec = self.sigmaVec
        g = self.gamma
        xval = self.xval
        xtrain = self.xtrain
        Yval = self.yval
        Ytrain = self.ytrain
        Ntrain = self.N
        Nval = len(Yval)
        
        self.getAlpha()
        a = self.alpha
        Kval = self.getK(xtrain,xval)
        Ktrain = self.getK(xtrain,xtrain)

        dKtrainds = np.zeros(shape=Ktrain.shape)
        dKvalds = np.zeros(shape=Kval.shape)

        for i in range(Ntrain):
            for j in range(Ntrain):
                dKtrainds[j,i] = self.dKds(xtrain[i],xtrain[j],sVec[i])
            for j in range(Nval):
                dKvalds[j,i] = self.dKds(xtrain[i],xval[j],sVec[i])
                
        # print('dKtrainds.shape =  %s' %((dKtrainds.shape,)))
        # print('dKvalds.shape =  %s' %((dKvalds.shape,)))

        
        
        p = -2/Nval * Kval.T@(Yval-Kval@a)@np.linalg.inv(2/Ntrain *Ktrain + g*np.eye(Ktrain.shape[0]))
        dEvalds = np.array([-1/Nval*( (a[i]*dKvalds[:,i]).T@(Yval-Kval@a) + (Yval-Kval@a).T@(a[i]*dKvalds[:,i])) for i in range(Ntrain)])
        
        # print('dEvalds.shape =  %s' %((dEvalds.shape,)))

        return dEvalds - np.array([p@(2/Ntrain*dKtrainds[:,i]*a[i]) for i in range(Ntrain)])

    
    def dHcdsigma(self):
        """
        """
        
        Kval = self.getK(self.xtrain,self.xval)
        KvalGrad = self.getKgrad(self.xtrain,self.xval)
        Kt = self.Kt
        KtGrad = self.getK(self.xtrain,self.xtrain)
        # print('Kt[0,3] =%g, sigma = %g' %(Kt[0,3],self.sigma))
        self.getAlpha()
        a = self.alpha
        g = self.gamma
        yv = self.yval
        yt = self.ytrain
        N = self.N
        Nval = len(yv)






        dEval = -1/Nval*Kval.T@(yv-Kval@a)
        d_dEval_ds = -1/Nval*( KvalGrad.T@(yv-Kval@a) - Kval.T@KvalGrad@a)

        dEin = -1/N*Kt.T@(yt-Kt@a) + g*Kt@a
        d_dEin_ds = -1/N*(KtGrad.T@(yt-Kt@a) - Kt.T@KtGrad@a) + g*KtGrad@a

        return -dEval@d_dEin_ds # -d_dEval_ds@dEin
        
    def getError(self,xtest,ytest):
        """
        """
        K = self.getK(self.xtrain,xtest)
        return (ytest-K@self.alpha).T@(ytest-K@self.alpha)

    def getHc(self):
        """
        """
        Kval = self.getK(self.xtrain,self.xval)
        KvalGrad = self.getKgrad(self.xtrain,self.xval)
        Kt = self.Kt
        KtGrad = self.getK(self.xtrain,self.xtrain)
        a = self.alpha
        g = self.gamma
        
        yv = self.yval
        yt = self.ytrain
        
        N = self.N
        Nval = len(yv)
        p = -1/Nval*Kval.T@(yv-Kval@a)
        tmp = -p@(-Kt.T@(yt-Kt@a) + g*Kt@a)
        print(-p@(-Kt.T@(yt-Kt@a))/N)
        print(-p@(g*Kt@a))
        return tmp/N
    










