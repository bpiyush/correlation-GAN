import numpy as np
import pandas as pd
from scipy.stats import pearsonr, entropy, chi2_contingency
import editdistance


def corr( X, Y ):
    return np.array([ pearsonr( x, y )[0] for x,y in zip( X.T, Y.T) ] )


def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n

    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)

    val = np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

    return val


def compute_cramers_v_matrix(X):

    dim = X.shape[1]
    mat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            mat[i][j] = cramers_v(X[:, i], X[:, j])

    return mat


def round_of_rating(number):
    return np.round(number * 2) / 2


def batchwise_editdistance(actual, predicted):
    
    actual = actual.astype(str)
    predicted = predicted.astype(str)

    nrows, ncols = actual.shape
    distances = np.zeros(nrows)
    for i in range(nrows):
        value = 0.0
        for j in range(ncols):
            value += editdistance.eval(actual.iloc[i, j], predicted.iloc[i, j]) / ncols
        distances[i] = value
    return distances




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
