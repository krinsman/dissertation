import numpy as np

# entrywise L1 -- less susceptibility to outliers, easier (imo) to interpret in this context, etc. etc.
def normalize(matrix):
    entrywise_l1_norm = np.linalg.norm(np.reshape(matrix, -1), ord=1)
    if entrywise_l1_norm == 0:
        return matrix
    else:
        return matrix / entrywise_l1_norm


def relative_error(truth, approx):
    return np.linalg.norm(
        np.reshape(normalize(truth) - normalize(approx), -1), ord=1
    ) / (np.linalg.norm(np.reshape(normalize(truth), -1), ord=1))
    # feel pretty confident the denominator is redundant but whatever I'm lazy


def entrywise_average(*coefficient_matrices, normalize_matrices=True):
    # ensure they all actually have the same shape
    assert (
        len(
            set(
                [
                    coefficient_matrix.shape
                    for coefficient_matrix in coefficient_matrices
                ]
            )
        )
        == 1
    )
    # otherwise this code would not work
    numerator = np.zeros((coefficient_matrices[0].shape))
    denominator = np.zeros(numerator.shape)

    for coefficient_matrix in coefficient_matrices:
        if normalize_matrices == True:
            numerator += normalize(
                coefficient_matrix
            )  # have some thoughts on why this might make more sense
        elif normalize_matrices == False:
            numerator += coefficient_matrix
        # assumes 0's correspond to missing values, since they usually do
        denominator += coefficient_matrix != 0

    # entry in denominator should be 0 only if estimate missing from all things
    # so want to divide 0 by 1, instead of trying to divide 0 by 0
    denominator += denominator == 0
    # this is similar to the V 1 thing in denominator of definitions of FDR, for example

    # Hadamard i.e. entry-wise division
    return numerator / denominator
