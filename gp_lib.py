import numpy as np
import scipy.spatial.distance as sdist
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt

RNG = np.random.default_rng()


class Scale():
    # standardizes 
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


class Kern():
    # Kernel template class
    nvar: float

    def k(*args):
        raise NotImplementedError

    def optim(self, X, y, init_k, mode='min', **kwargs):

        x_sc = Scale(X)
        y_sc = Scale(y)

        match mode:
            case 'min':
                opt_ = opt.minimize(
                    self.neg_likelihood, init_k, (x_sc.transform(X), y_sc.transform(y), self.nvar))
                return opt_
            case 'gridsearch':
                import optimizer
                opt_ = optimizer.grid_params(self, x_sc.transform(
                    X), y_sc.transform(y), self.nvar, **kwargs)
            case 'rand_start':
                # import optimizer
                n_trials = kwargs['n_trials']
                init_bounds = kwargs['init_bounds']
                opt_bounds = kwargs['opt_bounds']

                result = []

                for k in range(n_trials):

                    init_k = [RNG.uniform(*np.log(bound))
                              for bound in init_bounds]
                    try:
                        opt_ = opt.minimize(self.neg_likelihood, init_k, (x_sc.transform(
                            X), y_sc.transform(y), self.nvar), bounds=opt_bounds)
                        result.append([self.get_params(), opt_.fun])
                    except:
                        pass

                result.sort(key=lambda x: x[1])
                return result

                # opt_ = optimizer.grid_params(self, x_sc.transform(X), y_sc.transform())
                pass

        return None
        # return opt_

    def get_params(self):
        raise NotImplementedError

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

    def get_params(self):
        return self.scale, self.width


class Periodic(Kern):
    scale: float
    width: float
    period: float

    def __init__(self, nvar, scale, width, period) -> None:
        self.nvar = nvar
        self.scale = scale
        self.width = width
        self.period = period

    def k(self, dist, *kargs):
        if len(kargs):
            kargs = np.exp(kargs)
            self.scale, self.width, self.period = kargs

        return self.scale * np.exp(-2/self.width**2 * np.sin(np.pi/self.period*np.abs(dist))**2)

    def get_params(self):
        return self.scale, self.width, self.period


class PeriodicRbfAnd(Kern):
    per: Periodic
    rbf: Rbf

    def __init__(self, nvar, rbf_scale, rbf_width, per_width, per_period) -> None:
        self.nvar = nvar
        self.rbf = Rbf(nvar, rbf_scale, rbf_width)
        self.per = Periodic(nvar, 1., per_width, per_period)

    def k(self, dist, *kargs):
        if len(kargs):
            return self.rbf.k(dist, *kargs[:2]) * self.per.k(dist, *[0., *kargs[2:]])
        else:
            return self.rbf.k(dist) * self.per.k(dist)

    def get_params(self):
        return *self.rbf.get_params(), *self.per.get_params()[1:]


class PeriodicRbfOr(Kern):
    per: Periodic
    rbf: Rbf

    def __init__(self, nvar, rbf_scale, rbf_width, per_scale, per_width, per_period) -> None:
        self.nvar = nvar
        self.rbf = Rbf(nvar, rbf_scale, rbf_width)
        self.per = Periodic(nvar, per_scale, per_width, per_period)

    def k(self, dist, *kargs):
        if len(kargs):
            pass

        return self.rbf.k(dist, *kargs[:2]) + self.per.k(dist, *kargs[2:])

    def get_params(self):
        return *self.rbf.get_params(), *self.per.get_params()


class Gp():
    # Gaussian process solver

    kern: Kern
    nvar: float
    x_sc: Scale
    y_sc: Scale
    fitted: bool = False

    def __init__(self, kern, nvar) -> None:
        self.kern = kern
        self.nvar = nvar

    def _gp(self, X, y, Xt):
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
        return posterior_y, posterior_var

    def fit_predict(self, X, y, Xt):
        self.fit_std(X, y)
        return self.predict(X, y, Xt)

    def fit_std(self, X, y):
        x_sc = Scale(X)
        y_sc = Scale(y)

        self.x_sc = x_sc
        self.y_sc = y_sc

        self.fitted = True

    def standardize(self, X, y, Xt):
        X = self.x_sc.transform(X)
        y = self.y_sc.transform(y)
        Xt = self.x_sc.transform(Xt)
        return X, y, Xt

    def unstandardize(self, y, y_var):
        return self.y_sc.reverse(y), self.y_sc.std**2*y_var

    def predict(self, X, y, Xt):
        if not self.fitted:
            raise Exception('Not standardized')

        X, y, Xt = self.standardize(X, y, Xt)
        yt, yt_var = self._gp(X, y, Xt)
        return self.unstandardize(yt, yt_var)


def cov(A, B, k, *kargs):
    return k(dist(A, B), *kargs)


def dist(a, b):
    return sdist.cdist(np.expand_dims(a, 1), np.expand_dims(b, 1))


class DataStore():
    X: np.array
    Xt: np.array
    y: np.array
    yt: np.array

    def __init__(self) -> None:
        # init the training data

        df = pd.read_csv('extracted_data.csv')

        sub_df = df[['Time (hours)', 'Tide height (m)']]

        mask = pd.notna(sub_df['Tide height (m)'])

        
        test = sub_df.to_numpy()
        train = sub_df[mask].to_numpy()

        X, y = train[:, 0], train[:, 1]
        Xt = test[:, 0]

        yt = df['True tide height (m)'].to_numpy()
        df.loc[mask, 'True tide height (m)'] = np.nan

        self.X, self.y = X, y
        self.Xt, self.yt = Xt, yt

        self.mask = mask
        self.df = df

    def prepare_df(df):
        # preprocess dataset

        df['Time (hours)'] = pd.to_numeric(pd.to_datetime(df['Reading Date and Time (ISO)']))/int(1e9)
        df['Time (hours)'] -= df.loc[0, 'Time (hours)']
        df['Time (hours)'] /= 3600

    def load_dataset(df):
        # load and preprocess 

        df = pd.read_csv('sotonmet.txt')
        DataStore.prepare_df(df)
        df[['Time (hours)', 'Tide height (m)', 'True tide height (m)', 'Air temperature (C)', 'True air temperature (C)' ]].drop_duplicates(subset='Time (hours)').to_csv('extracted_data.csv', index=False)

    def apply_result(self, yt, cov_yt):
        # update the dataframe to include the predictions

        self.df['Estimated tide height (m)'] = yt
        self.df['Tide error (m)'] = 2*np.sqrt(cov_yt)


def test_gain(y, y_pred, var_pred):
    # return the log-likelihood of the estimated data
    
    res = [x for x in zip(y,y_pred,var_pred) if not np.isnan(x[0])]
    y, y_pred, var_pred = [np.array(u) for u in zip(*res)]


    logp = -1/2 * np.sum(np.log(2*np.pi*var_pred) +
                         np.power(y - y_pred, 2)/var_pred)
    return logp


def plot(df, title=None, savename=None, gt_alt=None):
    # plot the GP predictions for the current dataset.


    x_c = 'Time (hours)'
    y_c = 'Estimated tide height (m)'
    y_err = 'Tide error (m)'

    fig = plt.figure()

    x = df[x_c]
    mu = df[y_c]
    low = mu - df[y_err]
    up = mu + df[y_err]

    line_1, = plt.plot(x, mu, 'b-', label='Estimation')
    fill_1 = plt.fill_between(x, low, up, color='b', alpha=0.2)

    
    if gt_alt is None:
        scatter = plt.scatter(x,df['Tide height (m)'], label='Readings', s=1.5)
        real_ = plt.scatter(x, df['True tide height (m)'], label='Missing values', s=1.5)
    else:
        x_orig = gt_alt['Time (hours)']
        tmp_mask = pd.notna(gt_alt['Tide height (m)'])
        missing = gt_alt['True tide height (m)']
        missing[tmp_mask] = np.nan
        sensors = gt_alt['Tide height (m)']

        scatter = plt.scatter(x_orig, sensors, label='Readings', s=1.5)
        real_ = plt.scatter(x_orig, missing, s=1.5, label='Missing values')

    


    plt.margins(x=0)

    plt.legend()
    plt.xlabel('Time (hours)')
    plt.ylabel('Tide height (m)')

    plt.ylim([1, 5])

    if title is not None:
        plt.title(title)

    if savename is not None:
        plt.savefig(savename)

    plt.show()


def rbf_jacobian(params, alpha: np.array, L: np.array, dist: np.array, cov: np.array)-> np.array:
    # compute the jacobian of the RBF kernel
    var_, len_ = params
    div_K_wrt_L = 1/len_**3*dist**2 * cov
    div_K_wrt_var = 1/var_*cov
    return [1/2*alpha.T @ div @ alpha - 1/2 * np.trace(np.solve(L.T, np.solve(L, div))) for div in [div_K_wrt_var, div_K_wrt_L]]


def ar_gp_prediction(data: pd.DataFrame, kern):
    # predict the next value autoregressively, for the full sequence

    x_c = 'Time (hours)'
    y_c = 'Tide height (m)'

    predictions = []

    gp = Gp(kern, kern.nvar)
    tmp = data.loc[:, [x_c, y_c]].dropna().to_numpy()
    Xg, yg = tmp[:, 0], tmp[:, 1]
    gp.fit_std(Xg,yg)

    for k in range(len(data) - 1):
        train_df = data.loc[:k, [x_c, y_c]].dropna().to_numpy(dtype=np.float64)
        X, y = train_df[:, 0], train_df[:, 1]

        if not len(X):
            continue

        Xt = data.loc[k+1, [x_c]].to_numpy(dtype=np.float64)
        
        yt, yvar = gp.predict(X,y,Xt)

        predictions.append((*Xt,*yt, *yvar))

    return predictions

def plot_lookahead( predictions, savename):

    plot_df = pd.DataFrame({k:v for k,v in zip(['Time (hours)','Estimated tide height (m)', 'Tide error (m)'], zip(*predictions))})
    plot_df['Tide error (m)'] = 2*np.sqrt(plot_df['Tide error (m)'])

    orig_df = pd.read_csv('sotonmet.txt')
    DataStore.prepare_df(orig_df)

    plot(plot_df, savename=savename, gt_alt=orig_df)

