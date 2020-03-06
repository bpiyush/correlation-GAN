import numpy as np
from scipy.stats import entropy


def convert_to_prob_dist(X1, num_bins=100):
    ## convert to np.array
    X1 = np.asarray(X1)

    if len(X1.shape) == 1:
        X1 = X1.reshape(1, -1)

    P_X1 = np.zeros((num_bins, X1.shape[1]))
    for d in range(X1.shape[1]):
        P_X1[:, d] = np.histogram(X1[:, d], density=True, bins=num_bins)[0]

    return P_X1


def js_divergence(X1, X2, base=np.e, return_featurewise=False):
    '''
        Implementation of pairwise `jsd` based on
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    P_X1 = convert_to_prob_dist(X1)
    P_X2 = convert_to_prob_dist(X2)

    ## normalize p, X2 to probabilities
    P_X1, P_X2 = P_X1/P_X1.sum(axis=0), P_X2/P_X2.sum(axis=0)

    m = (1. / 2) * (P_X1 + P_X2)

    featurewise_jsd = entropy(P_X1, m, base=base) / 2. +  entropy(P_X2, m, base=base) / 2.
    jsd = np.mean(featurewise_jsd)

    if return_featurewise:
        return jsd, featurewise_jsd
    else:
        return jsd


def kl_divergence(X1, X2, base=np.e, return_featurewise=False):
    '''
        Implementation of pairwise `kld`
    '''
    P_X1 = convert_to_prob_dist(X1)
    P_X2 = convert_to_prob_dist(X2)

    ## normalize p, X2 to probabilities
    P_X1, P_X2 = P_X1/P_X1.sum(axis=0), P_X2/P_X2.sum(axis=0)

    featurewise_kld = entropy(P_X1, P_X2, base=base)
    kld = np.mean(featurewise_kld)

    if return_featurewise:
        return kld, featurewise_kld
    else:
        return kld


def reconstruction_error(X1, X2, return_featurewise=False):

    X1, X2 = np.asarray(X1), np.asarray(X2)

    if len(X1.shape) == 1:
        X1 = X1.reshape(1, -1)

    if len(X2.shape) == 1:
        X2 = X2.reshape(1, -1)

    featurewise_recs = np.linalg.norm(X1 - X2, axis=0)
    recs = np.mean(featurewise_recs)

    if return_featurewise:
        return recs, featurewise_recs
    else:
        return recs
