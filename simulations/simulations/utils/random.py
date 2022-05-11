def random_copy_numbers(length, rng, maximum_copy_number=15):
    return rng.integers(low=1, high=maximum_copy_number + 1, size=length)


import numpy as np
from scipy.stats import rankdata


def corrcoef(x, y):
    try:
        assert (x.size > 0) and (y.size > 0)  # check vectors aren't empty
        x = x - np.mean(x)  # center the first vector
        y = y - np.mean(y)  # center the second vector
        x_variance = np.dot(x, x) # technically this is off from variance by constant factor
        assert x_variance != 0.0  # check x is not constant
        y_variance = np.dot(y, y) # ditto the above comment
        assert y_variance != 0.0  # check y is not constant
        covariance = np.dot(x, y) # off from covariance by constant factor
        return covariance / np.sqrt(x_variance * y_variance)
    except AssertionError:
        return 0.0  # report correlation as 0, since that
    # implies correct conclusion that one is useless
    # for predicting other. (Rather than returning nan
    # and throwing warning like SciPy's function does.)
    # Maybe the correlation is technically undefined
    # in that case, but (to be blunt) I don't care.


def pearson(array1, array2):
    assert (
        array1.shape == array2.shape
    ), "Shape of the two arrays should be directly comparable. Reshape one if necessary to be unambiguous."
    return corrcoef(array1.ravel(), array2.ravel())


def spearman(array1, array2):
    # means redundant check in pearson, but will always pass for pearson because rankdata ravels
    assert (
        array1.shape == array2.shape
    ), "Shape of the two arrays should be directly comparable. Reshape one if necessary to be unambiguous."
    return pearson(
        rankdata(array1, method="average"), rankdata(array2, method="average")
    )


def covariance(array1, array2):
    assert (
        array1.shape == array2.shape
    ), "Shape of the two arrays should be directly comparable. Reshape one if necessary to be unambiguous."
    x = array1.ravel()
    y = array2.ravel()
    x = x - np.mean(x)
    y = y - np.mean(y)
    # Note that the 1/n terms cancel in numerator and denominator of correlation, making it redundant
    return np.dot(x, y) / np.size(x)


def rank_covariance(array1, array2):
    assert (
        array1.shape == array2.shape
    ), "Shape of the two arrays should be directly comparable. Reshape one if necessary to be unambiguous."
    return covariance(
        rankdata(array1, method="average"), rankdata(array2, method="average")
    )

from .generalized_jaccard import get_positive_part, get_negative_part

def mixed_sign_spearman(matrix1, matrix2):
    pos_matrix1 = get_positive_part(matrix1)
    pos_matrix2 = get_positive_part(matrix2)
    
    neg_matrix1 = get_negative_part(matrix1)
    neg_matrix2 = get_negative_part(matrix2)
    
    rank_vectors1 = [rankdata(pos_matrix1.ravel()), rankdata(neg_matrix1.ravel())]
    rank_vectors2 = [rankdata(pos_matrix2.ravel()), rankdata(neg_matrix2.ravel())]
    
    return pearson(np.concatenate(rank_vectors1), np.concatenate(rank_vectors2))