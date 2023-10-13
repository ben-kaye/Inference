from gp_lib import *
import numpy as np

import itertools


def get_param_combs(width, n_params):
    basis = np.linspace(-4, 4, 5)
    perms = list(itertools.product(basis, basis))
    return perms

def grid_params(kern: Kern, X, y, nvar, dim, n_p, bounds):
    # basis = 
    perms = list(itertools.product(*[np.linspace(*np.log(bound), n_p) for bound in bounds]))

    nlogps = [kern.neg_likelihood(params, X, y, nvar) for params in perms]
    
    max_i = np.argmin(nlogps)

    return nlogps, nlogps[max_i], np.exp(perms[max_i])

if __name__ == '__main__':
    data = DataStore()

    x_sc = Scale(data.X)
    y_sc = Scale(data.y)

    nvar = 1e-1


    kern = PeriodicRbfOr(nvar, *[0.31622777, 0.03162278, 0.31622777, 1.        , 0.34641016])

    logps, best_val, best_k_params = grid_params(kern, x_sc.transform(data.X), y_sc.transform(data.y), nvar, dim=5, n_p=5, bounds=[(1e-2, 1 ), (1e-2, 1,), (1e-2, 1), (1e-2, 1), (0.3, 0.4)])

    pass

    
    
