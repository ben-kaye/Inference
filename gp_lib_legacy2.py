import numpy as np
import scipy as sp
import pandas as pd

# py 3.12


class MercerKernel():
    def k(self, u, v):
        pass

    def K(self, *params):
        pass

    def get(self, X):
        return self.K(self.get_params(X))

    def get_params(self):
        pass


class RbfKern(MercerKernel):
    var: float
    len: float
    nvar: float
    jitter: float

    def __init__(self, var, len, nvar, jitter=1e-6) -> None:
        super().__init__()
        self.var = var
        self.len = len
        self.nvar = nvar
        self.jitter = jitter

    def k(self, u, v):
        return self.var*np.exp(-1/2/self.len**2 * (u - v)**2)

    # apply k to matrix by element
    def K(self, Xprime):

        res = self.var*np.exp(-1/2/self.len**2 * Xprime)
        if Xprime.shape[0] == Xprime.shape[1]:  # include diagonal noise
            res += (self.nvar + self.jitter)*np.eye(Xprime.shape[-1])
        return res


class Regressor():
    X: np.array
    y: np.array

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, Xt, **kwargs):
        raise NotImplementedError


class GaussianProcess(Regressor):
    kern: MercerKernel
    cov: np.array
    fitted: bool = False

    def __init__(self, kern=RbfKern(nvar=0.04, len=0.4, var=1)) -> None:
        super().__init__()
        self.kern = kern

    def fit(self, X, y, jitter=1e-6):
        # create covariance matrix
        self.X = X
        self.y = y

        self.cov = self.kern.K(make_diff(X, X))

        self.fitted = True
        pass

        return self

    def predict(self, Xt):
        if not self.fitted:
            raise Exception()

        covstar = self.kern.K(make_diff(Xt, self.X))
        covdstar = self.kern.K(make_diff(Xt, Xt))
        map_ = covstar @ np.linalg.solve(self.cov, self.y)
        var_ = covdstar - covstar @ np.linalg.solve(self.cov, covstar.T)
        return map_, np.diag(var_)


class Scalar(Regressor):
    X_std: float
    X_loc: float
    y_std: float
    y_loc: float

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y, **kwargs):
        self.X_std = np.std(X)
        self.y_std = np.std(y)

        self.X_loc = np.mean(X)
        self.y_loc = np.mean(y)

        return self

    def transform(self, X, y):
        return (X-self.X_loc)/self.X_std, (y - self.y_loc)/self.y_std
    
    def reverse(self, X,y):
        return X*self.X_std + self.X_loc, y*self.y_std + self.y_loc


def make_diff(X1, X2):
    m = X1.shape[-1]
    n = X2.shape[-1]

    return np.array(
        [[X1[u] - X2[v] for v in range(n)] for u in range(m)]
    )
