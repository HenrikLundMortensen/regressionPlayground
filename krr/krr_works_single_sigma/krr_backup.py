from krr_util import *

class regressor():

    def __init__(self,xtrain,ytrain,kernel,sigma=1,gamma=1,xval=None,yval=None):
        """

        """
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.N = len(ytrain)
        self.kernel = kernel
        self.sigma = sigma
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
        s = self.sigma
        K = np.array([[k(xt[i],x[j],s) for i in range(len(xt))] for j in range(len(x))])
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

        self.alpha = np.linalg.inv(2*self.Kt/N+g*np.eye(self.Kt.shape[0]))@yt

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

        p = 2/(Ntrain)*Kval.T@(Yval - Kval@a) @ np.linalg.inv(s*Ktrain - 2/(Ntrain)*Ktrain.T@Ktrain)
        dEvalds  = -1/(Nval)*(  a.T@dKvalds.T@(Yval - Kval@a) + (Yval-Kval@a).T@dKvalds@a)

        return dEvalds - p@(2/Ntrain*(dKtrainds.T@(Ytrain - Ktrain@a) - Ktrain.T@dKtrainds@a   ) + s*dKtrainds@a )

        

    
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
        self.getAlpha()
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
    










