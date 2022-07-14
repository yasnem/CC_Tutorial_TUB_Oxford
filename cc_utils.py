import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as sklearn_kernel
from sklearn.metrics import euclidean_distances


def cholesky_decomposition(K):
    K = (K.T + K) / 2 + np.eye(K.shape[0]) * 10e-5
    assert np.all(K.T == K), "Kernel matrix is not symmetric"
    L = np.linalg.cholesky(K)
    return L


def compute_gram_matrix(x, param):
    """Computes kernel gram matrix of data x using median heuristic and rbf kernel"""
    if param['kernel'] == 'rbf':
        if len(x.shape) < 2:
            x = np.expand_dims(x, axis=1)
        kernel_width, _ = median_heuristic(x, x)
        kernel_matrix = rbf_kernel(x, x, kernel_width)
    elif param['kernel'] == 'linear':
        kernel_matrix = linear_kernel(x, x, param['kernel_param'])
    else:
        raise ValueError
    return kernel_matrix


def mmd_eps(n_sample, alpha=0.05, ckern=1.0):
    '''compute the mmd guarantee
    alpha = 0.05 # significance level
    ckern = 1.0 # constant due to kernel
    '''
    eps_opt = (1 + np.sqrt(2 * np.log(1 / alpha))) * np.sqrt(ckern / n_sample)
    return eps_opt


def median_heuristic(X, Y):
    '''
    the famous kernel median heuristic
    :param X:
    :param Y:
    :return:
    '''
    distsqr = euclidean_distances(X, Y, squared=True)
    kernel_width = np.sqrt(0.5 * np.median(distsqr))

    '''in sklearn, kernel is done by K(x, y) = exp(-gamma ||x-y||^2)'''
    kernel_gamma = 1.0 / (2 * kernel_width ** 2)

    return kernel_width, kernel_gamma


def compute_scenarionumber(beta, n_control_vars, p_chance):
    """
    Compute the min. number of scenarios for a given confidence level.

    For a given confidence level beta, probability for individual chance
    constraints and the number of control variables,
    compute the minimum number of scenarios according to
    Hewing, Zeilinger ICSL 2020.
    """
    N = 2 * ((n_control_vars - 1) * np.log(2) - np.log(beta)) / (1 - p_chance)
    return N


def rbf_kernel(x, y, sigma):
    """Wrapper for the RBF kernel from sklearn."""
    if len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)
    if len(y.shape) < 2:
        y = np.expand_dims(y, axis=1)
    gamma = 1 / (2 * sigma ** 2)
    K = sklearn_kernel(x, y, gamma)
    # K = (K + K.T) / 2
    # assert np.all(K == K.T), "Something is wrong here!{}".format(K)
    return K


def linear_kernel(x, y, c):
    if len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)
    if len(y.shape) < 2:
        y = np.expand_dims(y, axis=1)
    K = x @ y.T + c
    return K
