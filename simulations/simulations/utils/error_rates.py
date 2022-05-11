import numpy as np


def mixed_fnr(truth, estimate, positive_part=False, negative_part=False):
    assert (positive_part == False) or (
        negative_part == False
    ), "`positive_part` and `negative_part` are mutually exclusive; please choose one, the other, or neither to be `True`."

    if positive_part == True:
        neg_truth = np.zeros(truth.shape).astype(bool)
        neg_estimate = np.zeros(estimate.shape).astype(bool)
    else:
        neg_truth = truth < 0
        neg_estimate = estimate < 0

    if negative_part == True:
        pos_truth = np.zeros(truth.shape).astype(bool)
        pos_estimate = np.zeros(estimate.shape).astype(bool)
    else:
        pos_truth = truth > 0
        pos_estimate = estimate > 0

    pos_false_negatives = np.sum(pos_truth & (~pos_estimate))
    neg_false_negatives = np.sum(neg_truth & (~neg_estimate))

    return (pos_false_negatives + neg_false_negatives) / (
        # see note for FDR. admittedly Barber and Candes don't talk about FNR
        max(np.sum(pos_truth) + np.sum(neg_truth), 1)
    )


def mixed_fdr(truth, estimate, positive_part=False, negative_part=False):
    assert (positive_part == False) or (
        negative_part == False
    ), "`positive_part` and `negative_part` are mutually exclusive; please choose one, the other, or neither to be `True`."

    if positive_part == True:
        neg_truth = np.zeros(truth.shape).astype(bool)
        neg_estimate = np.zeros(estimate.shape).astype(bool)
    else:
        neg_truth = truth < 0
        neg_estimate = estimate < 0

    if negative_part == True:
        pos_truth = np.zeros(truth.shape).astype(bool)
        pos_estimate = np.zeros(estimate.shape).astype(bool)
    else:
        pos_truth = truth > 0
        pos_estimate = estimate > 0

    pos_false_positives = np.sum(pos_estimate & (~pos_truth))
    neg_false_positives = np.sum(neg_estimate & (~neg_truth))

    return (pos_false_positives + neg_false_positives) / (
        # following Barber and Candes 2015 https://arxiv.org/pdf/1404.5609.pdf
        # because appeal to authority and couldn't think of any more
        # sensible convention when sum in denominator equals zero
        max(np.sum(pos_estimate) + np.sum(neg_estimate), 1)
    )
