# Define the DRR model class
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.kernel_ridge import KernelRidge

class DRR(BaseEstimator, TransformerMixin):
   
    def __init__(self,estimator_type="Poly"):
        # regressor
        if estimator_type=="Poly":        
            self.parameters = {'polynomialfeatures__degree':np.arange(1,7), 'ridge__alpha':[0.01,0.1,1,10]}
            self.model = make_pipeline(PolynomialFeatures(), Ridge())
        elif estimator_type=="KRR":        
            self.parameters = {'kernel':('linear', 'rbf'), "alpha": [1e0, 0.1, 1e-2, 1e-3],"gamma": np.logspace(-2, 2, 5)}
            self.model = KernelRidge()


    def fit_transform(self,X):
        # PCA
        self.m = np.mean(X, axis = 0)
        Xm = X - self.m[np.newaxis,:]
        cov_matrix = np.dot(Xm.T, Xm) / Xm.shape[0]
        evals, evecs = np.linalg.eigh(cov_matrix)   
        #ordenamos
        idx = np.argsort(-evals)
        self.evecs = evecs[:,idx]
        evals = evals[idx]
        #aplicamos
        Xpca = np.dot(Xm,self.evecs)

        # DRR
        Xdrr = Xpca.copy()

        self.models = []

        for n in np.arange(X.shape[1]-1,0,-1):
            
            clf = GridSearchCV(self.model, self.parameters)
            clf.fit(Xpca[:,0:n], Xpca[:,n])
            x_hat = clf.predict(Xpca[:,0:n])

            Xdrr[:,n] = Xpca[:,n]-x_hat
            self.models.append(clf)
            print(n)

        return Xdrr, Xpca

    def transform(self,X):
        # PCA
        Xm = X - self.m[np.newaxis,:]
        #aplicamos
        Xpca = np.dot(Xm,self.evecs)

        # DRR
        Xdrr = Xpca.copy()

        for n in np.arange(X.shape[1]-1,0,-1):
            ind_m = X.shape[1]-n-1
            x_hat = self.models[ind_m].predict(Xpca[:,0:n])
            Xdrr[:,n] = Xpca[:,n]-x_hat
            print(n)

        return Xdrr  

    def transform_pca(self,X):
        # PCA
        Xm = X - self.m[np.newaxis,:]
        #aplicamos
        Xpca = np.dot(Xm,self.evecs)    

        return Xpca

    def inverse(self,Xdrr):
        inv_Xdrr = Xdrr.copy()
        for n in np.arange(Xdrr.shape[1]-2,-1,-1):
            x_hat = self.models[n].predict(inv_Xdrr[:,0:Xdrr.shape[1]-n-1])
            inv_Xdrr[:,Xdrr.shape[1]-n-1] = inv_Xdrr[:,Xdrr.shape[1]-n-1] + x_hat
            print(n)  
        # PCA inv
        Xm_inv = np.dot(inv_Xdrr,self.evecs.T)
        X_inv = Xm_inv + self.m[np.newaxis,:]

        return X_inv

    def inverse_pca(self,Xpca):
        inv_Xpca = Xpca.copy()
        # PCA inv
        Xm_inv = np.dot(inv_Xpca,self.evecs.T)
        X_inv = Xm_inv + self.m[np.newaxis,:]

        return X_inv
