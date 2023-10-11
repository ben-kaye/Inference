import numpy as np
import scipy.spatial.distance as sdist
import scipy.optimize as opt
import pandas as pd


class Scale():
    loc: float
    std: float

    def __init__(self, v) -> None:
        self.loc = np.mean(v)
        self.std = np.std(v)

    def transform(self, v):
        return (v - self.loc)/self.std

    def reverse(self, v):
        return self.std*v + self.loc


def cholesky_solve(A, b):
    # returns A\b, lower triang cholesky
    L = np.linalg.cholesky(A)
    return np.linalg.solve(L.T, np.linalg.solve(L, b)), L


def torch_optim():
    pass


def dist(a, b):
    return sdist.cdist(np.expand_dims(a, 1), np.expand_dims(b, 1))


class Kern():
    nvar: float

    def k(*args):
        raise NotImplementedError

    def optim(self, X, y, init_k):
        opt_ = opt.minimize(self.neg_likelihood, init_k, (X, y, self.nvar))
        return opt_

    def neg_likelihood(self, kargs, X, y, nvar):
        data_cov = cov(X, X, self.k, *kargs) + nvar*np.eye(X.shape[-1])

        alpha, L = cholesky_solve(data_cov, y)
        logdet = 2*np.sum(np.log(np.diag(L)))

        n = y.shape[-1]
        logp = -1/2 * y.T @ alpha - 1/2 * logdet - n/2 * np.log(2*np.pi)
        return -logp


class Rbf(Kern):
    scale: float
    width: float

    def __init__(self, nvar, scale, width) -> None:
        self.nvar = nvar
        self.scale = scale
        self.width = width

    def k(self, dist, *kargs):
        if len(kargs):
            kargs = np.exp(kargs)
            self.scale, self.width = kargs
        return self.scale * np.exp(-1/2/self.width**2 * np.power(dist, 2))


class Gp():
    kern: Kern
    nvar: float
    x_sc: Scale
    y_sc: Scale

    def __init__(self, kern, nvar) -> None:
        self.kern = kern
        self.nvar = nvar

    def fit_predict(self, X, y, Xt):
        x_sc = Scale(X)
        y_sc = Scale(y)

        X = x_sc.transform(X)  # + 1e-6*np.ones_like(X)
        y = y_sc.transform(y)

        Xt = x_sc.transform(Xt)

        self.x_sc = x_sc
        self.y_sc = y_sc

        test_cov = cov(Xt, Xt, self.kern.k)
        mixed_cov = cov(Xt, X, self.kern.k)
        data_cov = cov(X, X, self.kern.k) + self.nvar*np.eye(X.shape[-1])

        alpha, L = cholesky_solve(data_cov, y)

        self.L = L
        self.alpha = alpha

        posterior_y = mixed_cov @ alpha
        nu = np.linalg.solve(L, mixed_cov.T)
        posterior_cov = test_cov - nu.T @ nu

        posterior_var = np.diag(posterior_cov)

        posterior_y = y_sc.reverse(posterior_y)
        posterior_var = y_sc.std**2 * posterior_var

        return posterior_y, posterior_var


def cov(A, B, k, *kargs):
    return k(dist(A, B), *kargs)


class DataStore():
    X: np.array
    Xt: np.array
    y: np.array
    yt: np.array

    def __init__(self) -> None:
        data = pd.read_csv('extracted_data.csv')

        mask = pd.notna(data['Tide height (m)'])

        test = data[~mask].to_numpy()
        train = data[mask].to_numpy()

        X, y = train[:, 0], train[:, 1]
        Xt = test[:, 0]

        data['Inputed'] = False
        data.loc[~mask, 'Inputed'] = True

        src_df = pd.read_csv('sotonmet.txt')
        n_mask = pd.Series(
            [mask[k] if k in mask.index else True for k in range(len(src_df))])
        truth = pd.read_csv(
            'sotonmet.txt').loc[~n_mask, 'True tide height (m)'].to_numpy()
        self.X, self.y, self.Xt = X, y, Xt
        self.yt = truth

        self.data = data


def test_gain(y, y_pred, var_pred):
    logp = -1/2 * np.sum(np.log(2*np.pi*var_pred) +
                         np.power(y - y_pred, 2)/var_pred)
    return logp


if __name__ == '__main__':
    data = DataStore()

    nvar = 1e-1

    kern = Rbf(nvar, 0.5, 0.1)
    train_logp = kern.optim(data.X, data.y, [0.5, 0.1]).fun

    gp = Gp(kern, nvar)
    yt, yvar = gp.fit_predict(data.X, data.y, data.Xt)

    test_logp = test_gain(data.yt, yt, yvar)

    print(f'{train_logp=: .3e}, {test_logp=: .3e}')
