import numpy as np


def get_positive_part(signed_possibly_weighted_adjacency_matrix):
    # seems slightly faster for whatever reason than the more intuitive
    # network1_positive_part = signed_possibly_weighted_adjacency_matrix1 * (signed_possibly_weighted_adjacency_matrix1 > 0)
    # as a bonus this doesn't create matrices with negative zeros, unlike the above
    return (
        signed_possibly_weighted_adjacency_matrix
        + np.abs(signed_possibly_weighted_adjacency_matrix)
    ) / 2


def get_negative_part(signed_possibly_weighted_adjacency_matrix):
    # seems slightly faster for whatever reason than the more intuitive
    # network1_negative_part = (-signed_possibly_weighted_adjacency_matrix1) * (signed_possibly_weighted_adjacency_matrix1 < 0)
    # as a bonus this doesn't create matrices with negative zeros, unlike the above
    return (
        np.abs(signed_possibly_weighted_adjacency_matrix)
        - signed_possibly_weighted_adjacency_matrix
    ) / 2


def mixed_sign_Jaccard_similarity(
    signed_possibly_weighted_adjacency_matrix1,
    signed_possibly_weighted_adjacency_matrix2,
    unweighted=False,
    positive_part=False,
    negative_part=False,
):
    """Reduces to regular weighted Jaccard for non-negative only adjacency matrices.
    Unweighted version also reduces to regular unweighted Jaccard for non-negative only adjacency matrices.
    Reduces to unweighted version when applied to sign matrix of original weighted matrix.
    Positive part and negative part always have non-negative adjacency matrices, so their
    regular Jaccard similarities can be computed with this in light of the aforementioned reduction."""

    assert (positive_part == False) or (
        negative_part == False
    ), "Can't compute the Jaccard similarities for the positive and negative parts of the network at the same time -- please choose one or the other."

    if unweighted == True:
        # preserves signs, removes weights
        signed_possibly_weighted_adjacency_matrix1 = np.sign(
            signed_possibly_weighted_adjacency_matrix1
        )
        signed_possibly_weighted_adjacency_matrix2 = np.sign(
            signed_possibly_weighted_adjacency_matrix2
        )

    if positive_part == True:
        signed_possibly_weighted_adjacency_matrix1 = get_positive_part(
            signed_possibly_weighted_adjacency_matrix1
        )
        signed_possibly_weighted_adjacency_matrix2 = get_positive_part(
            signed_possibly_weighted_adjacency_matrix2
        )

    if negative_part == True:
        signed_possibly_weighted_adjacency_matrix1 = get_negative_part(
            signed_possibly_weighted_adjacency_matrix1
        )
        signed_possibly_weighted_adjacency_matrix2 = get_negative_part(
            signed_possibly_weighted_adjacency_matrix2
        )

    # yes, this function is obviously not the most efficient way to compute e.g. regular Jaccard of pos. or neg. parts
    network1_positive_part = get_positive_part(
        signed_possibly_weighted_adjacency_matrix1
    )
    network1_negative_part = get_negative_part(
        signed_possibly_weighted_adjacency_matrix1
    )

    network2_positive_part = get_positive_part(
        signed_possibly_weighted_adjacency_matrix2
    )
    network2_negative_part = get_negative_part(
        signed_possibly_weighted_adjacency_matrix2
    )

    # np.minimum, not np.min, for entrywise min
    numerator = np.sum(
        np.minimum(network1_positive_part, network2_positive_part)
    ) + np.sum(np.minimum(network1_negative_part, network2_negative_part))
    # analogously np.maximum, not np.max
    denominator = np.sum(
        np.maximum(network1_positive_part, network2_positive_part)
    ) + np.sum(np.maximum(network1_negative_part, network2_negative_part))

    return numerator / denominator


def mixed_sign_Jaccard_distance(
    signed_possibly_weighted_adjacency_matrix1,
    signed_possibly_weighted_adjacency_matrix2,
    unweighted=False,
    positive_part=False,
    negative_part=False,
):
    return 1.0 - mixed_sign_Jaccard_similarity(
        signed_possibly_weighted_adjacency_matrix1,
        signed_possibly_weighted_adjacency_matrix2,
        unweighted=unweighted,
        positive_part=positive_part,
        negative_part=negative_part,
    )
