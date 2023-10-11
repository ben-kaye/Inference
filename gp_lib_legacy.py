import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np


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


def gaussian_process_legacy(X, y, Xt, kernel_func, nvar=0.04):
    x_sc = Scale(X)
    y_sc = Scale(y)

    X = x_sc.transform(X)
    y = y_sc.transform(y)

    Xt = x_sc.transform(Xt)

    data_cov = apply_cov(X, X, kernel_func) + nvar*np.eye(X.shape[-1])
    test_cov = apply_cov(Xt, Xt, kernel_func)
    mixed_cov = apply_cov(Xt, X, kernel_func)

    post_y = np.matmul(mixed_cov, np.linalg.solve(data_cov, y))
    post_cov_y = test_cov - \
        np.matmul(mixed_cov, np.linalg.solve(data_cov, mixed_cov.T))

    post_var = np.diag(post_cov_y)

    return y_sc.reverse(post_y), y_sc.std**2*post_var


def gaussian_process(X, y, Xt, yt_truth, kernel_func, nvar):
    x_sc = Scale(X)
    y_sc = Scale(y)

    X = x_sc.transform(X)  # + 1e-6*np.ones_like(X)
    y = y_sc.transform(y)

    Xt = x_sc.transform(Xt)

    test_cov = apply_cov(Xt, Xt, kernel_func)
    mixed_cov = apply_cov(Xt, X, kernel_func)

    data_cov = apply_cov(X, X, kernel_func) + nvar*np.eye(X.shape[-1])

    # import matplotlib.pyplot as plt

    # plt.imshow(data_cov)
    # plt.show()

    # #

    chole = np.linalg.cholesky(data_cov)
    
    alpha = np.linalg.solve(chole.T, np.linalg.solve(chole, y))

    post_y = np.matmul(mixed_cov, alpha)
    nu = np.linalg.solve(chole, mixed_cov.T)

    post_cov_y = test_cov - np.matmul(nu.T, nu)

    post_var = np.diag(post_cov_y)

    real_y = y_sc.reverse(post_y)
    real_var = y_sc.std**2*post_var
    logp = perf_likelihood(yt_truth, real_y, real_var)

    return real_y, real_var, logp


class RBF():
    # nvar: float/
    var: float
    scale: float

    def __init__(self, var, scale) -> None:
        # self.nvar = nvar
        self.var = var
        self.scale = scale

    def k(self, a, b):
        return self.var * np.exp(-1/2/self.scale**2*(a-b)**2)


class Periodic():
    pass
    scale: float
    length: float
    period: float

    def __init__(self, scale, length, period) -> None:
        self.scale = scale
        self.length = length
        self.period = period

        pass

    def k(self, a, b):
        return self.scale*np.exp(-2/self.length**2 * np.sin(np.pi/self.period*np.abs(a - b))**2)


def apply_cov(A, B, func):
    m = A.shape[-1]
    n = B.shape[-1]
    return np.array([[
        func(A[u], B[v])
        for v in range(n)
    ] for u in range(m)])


def apply_cov_var(A, B, func, kern_args):
    m = A.shape[-1]
    n = B.shape[-1]
    return np.array([[
        func(A[u], B[v], kern_args)
        for v in range(n)
    ] for u in range(m)])


def get_scaled_data(X, y):
    x_sc = Scale(X)
    y_sc = Scale(y)

    X = x_sc.transform(X)
    y = y_sc.transform(y)

    return X, y


def opt_model_evidence(X, y):
    pass


class KernOpt():
    # params: list

    def k(a, b, *kargs):
        raise NotImplementedError()


class PeriodicRbf(KernOpt):
    var: float
    len_rbf: float
    len_per: float
    period: float

    def __init__(self, var, len_rbf, len_per, period) -> None:
        self.var = var
        self.len_rbf = len_rbf
        self.len_per = len_per
        self.period = period

    def k(self, a, b, *kargs):
        if len(kargs):
            kargs = np.exp(kargs[0])
            self.var, self.len_rbf, self.len_per, self.period = kargs

        return self.var*(np.exp(-2/self.len_per**2 * np.sin(np.pi/self.period*np.abs(a - b))**2) + np.exp(-1/2/self.len_rbf**2*(a-b)**2))


class RbfOpt(KernOpt):
    var: float
    len_: float

    def __init__(self, var, len_) -> None:
        self.var = var
        self.len_ = len_

    def k(self, a, b, *kargs):
        if len(kargs):
            kargs = np.exp(kargs[0])
            self.var, self.len_ = kargs
        self.var*np.exp(-1/2/self.len_**2*(a-b)**2)


def get_logp_opt(kern_args, X, y, kern, nvar):
    data_cov = apply_cov_var(X, X, kern, kern_args) + nvar*np.eye(X.shape[-1])
    L = np.linalg.cholesky(data_cov)

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
#
    logdet = 2*np.sum(np.log(np.diag(L)))

    # print('WARNING: not logging the det')
    # logdet = 2*np.sum(np.diag(L))

    n = y.shape[-1]

    logp = -1/2 * y.T @ alpha - 1/2 * logdet - n/2 * np.log(2*np.pi)

    return -logp


def optimise_kernel(X, y, kernel, nvar, kern_init, bounds, max_iters=10):
    import scipy.optimize
    resp = scipy.optimize.minimize(get_logp_opt, kern_init, args=(
        X, y, kernel.k, nvar), bounds=bounds, options=dict(maxiter=max_iters, disp=True))


def perf_likelihood(y, y_pred, var_pred):
    logp = -1/2 * np.sum(np.log(2*np.pi*var_pred) +
                         np.power(y - y_pred, 2)/var_pred)
    return logp


def plot(data):
    # px.scatter(data,)
    # go.Figure([
    # go.Scatter(x=X,y=y, mode='markers'), go.Scatter(x=Xt,y=yt, mode='markers'), go.Scatter(x=xtruth, y=ytruth, mode='markers')
    # ])
    x_c = 'Time (hours)'
    y_c = 'Tide height (m)'
    y_err = 'Tide error (m)'

    fig = plt.figure()

    x = data[x_c]
    mu = data[y_c]
    low = mu - data[y_err]
    up = mu + data[y_err]

    line_1, = plt.plot(x, mu, 'b-')
    fill_1 = plt.fill_between(x, low, up, color='b', alpha=0.2)

    real_ = plt.plot(x, data['Tide actual (m)'], 'r--')

    plt.margins(x=0)

    plt.legend([(line_1, fill_1), real_,], ['Predicted data', 'Ground truth'])
    plt.show()
    # return


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('extracted_data.csv')

    mask = pd.notna(data['Tide height (m)'])
    test = data[~mask].to_numpy()
    train = data[mask].to_numpy()
    X, y = train[:, 0], train[:, 1]
    Xt = test[:, 0]

    src_df = pd.read_csv('sotonmet.txt')
    n_mask = pd.Series(
        [mask[k] if k in mask.index else True for k in range(len(src_df))])
    truth = pd.read_csv(
        'sotonmet.txt').loc[~n_mask, 'True tide height (m)'].to_numpy()

    # kern = RBF(var=0.1, scale=0.1)

    kern = PeriodicRbf(0, 0, 0, 0)

    nvar = 9e-2

    kern_init = [-2.30258509, -3.21887582, -2.30258509, -1.03845837]
    bounds = [(-4, 2), (-4, 0), (-4, 0), (-1.2, -0.9)]

    # testp = optimise_kernel(X, y, kern, nvar, kern_init, bounds)

    yt, cov_yt, logp = gaussian_process(X, y, Xt, truth,  kern.k)
    # optimal vals: [kern.var, kern.len_rbf, kern.len_per, kern.period]
    # [0.5051306832635492, 1.0, 1.0, 0.4065696556749025]

    # print(f'{logp=}')

    data.loc[~mask, 'Tide height (m)'] = yt
    data['Inputed'] = False
    data.loc[~mask, 'Inputed'] = True

    data.rename(
        columns={'Reading Date and Time (ISO)': 'Time (hours)'}, inplace=True)
    data['Time (hours)'] /= 3600

    # import plotly.express as px
    # fig = px.scatter(data, x='Time (hours)', y='Tide height (m)',
    #                  color='Inputed', template='plotly_white')
    # fig.update_layout(
    #     title='RBF kernel, with variance=0.1std, length scale=0.1std', ).show()

    # fig.write_image('rbf.png')
