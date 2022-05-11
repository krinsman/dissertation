import numpy as np
import numpy.linalg as la


def get_lbp_matrix(matrix):
    """LBP = linearized belief propagation"""
    assert (len(matrix.shape) == 2) and (
        matrix.shape[0] == matrix.shape[1]
    ), "Input matrix needs to be square."
    dim = matrix.shape[0]
    D = np.diag(np.sum(matrix, axis=1))  # row sums, i.e. outdegree matrix
    # inf_norm = np.max(np.sum(np.abs(matrix), axis=1))
    inf_norm = la.norm(matrix, ord=np.inf)
    #     try:
    #         assert not np.isclose(inf_norm, 0.)
    #         # if it's close now, it'll be even closer after being squared
    #     except AssertionError:
    #         # If network is all 0's, i.e. completely disconnected
    #         # Then obviously Neumann series is also the all 0's matrix
    #         # for any possible choice of epsilon
    #         return np.zeros((dim, dim))
    #         # trying to use typical choice of epsilon leads to division by zero in next line
    epsilon = 1.0 / (2.0 + inf_norm)
    # epsilon = 1.0 / (1.0 + inf_norm)
    M1 = ((epsilon) / (1.0 - epsilon ** 2)) * matrix
    M2 = ((epsilon ** 2) / (1.0 - epsilon ** 2)) * D
    return la.inv(np.eye(dim) - M1 + M2)


signed_sqrt = lambda array: np.sign(array) * np.sqrt(np.abs(array))


def deltacon(matrix1, matrix2):
    """Assumes matrix1 and matrix2 represent adjacency matrices of possibly directed and possibly weighted networks, where weights can be positive or negative. (I.e. arbitrary matrices, but they have an interpretation.)"""
    S1 = get_lbp_matrix(matrix1)
    S2 = get_lbp_matrix(matrix2)
    return np.sqrt(np.sum((signed_sqrt(S1) - signed_sqrt(S2)) ** 2, axis=None))


# Consider the following matrix:
#      [[ 0. ,  0. ,  0. ,  0. ,  0. , -4.5,  0. ,  0. ,  0. ,  0. ],
#       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
#       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -3.5],
#       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
#       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
#       [-4.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
#       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
#       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
#       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
#       [ 0. ,  0. , -4.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]]
#
# Using epsilon = 1 / (1 + inf_norm)
# the resulting W = M1 - M2 matrix winds up having both
# infinity norm and spectral radius exactly equal to 1
# thus its Neumann series doesn't converge.
# Technically the proof only showed that using a value of
# epsilon strictly less than that was guaranteed to ensure
# convergence, but I didn't seriously expect there to exist
# matrices for which not only the infinity norm of W equaled one
# (that alone would be straightforward to construct) but for which
# the spectral radius also equaled the infinity norm (and one).
# In hindsight sparsity is the key to coming up with such a counterexample.
# But anyway I never expected I would ever randomly meet such an unlikely
# matrix "in the wild". (I did.) (Professor Koutra's code also seemed to imply
# not expecting to ever encounter an analogous such matrix for the Frobenius norm).
# Anyway such nearly garbage matrices seem to be where the DeltaCon distances
# of order 10^8 or whatever are coming from -- their similarity matrices
# have some really large entries because the corresponding W have spectral
# radii way too close to 1. And in this case exactly equal to 1 (yay!)
#
# So anyway I will use 1 / (2 + inf_norm) from now on since that is
# guaranteed to be less than 1 / (1 + inf_norm), thereby guaranteeing
# converging whatever. Also seems to cause the biggest difference from
# before for precisely those matrices that have W closest to being singular
# (or are singular) using the old definition and probably would best benefit
# from having a big shove away from singularity (to avoid similarity matrices
# with inordinately large values), while having the least effect on those
# whose W doesn't need a big shove away from singularity. Admittedly I'm not
# sure about this. Anyway in FaBP paper they claimed to show empirically that
# at least result of FaBP (in terms of classification accuracy) were largely
# insensitive to the specific value of hh (and thus epsilon = 2hh).

# Also bounding epsilon from above by 1/2 like this does (instead of 1)
# also means we should be able to avoid the division by zero errors too.
# So this is probably for the best (I hope).
