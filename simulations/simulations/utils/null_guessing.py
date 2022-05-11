import numpy as np
from scipy.stats import binom
from itertools import product
from tensorly.tenalg import outer as outer_product

# The FNR distributions are just scaled binomials,
# since the denominators are deterministic
def compute_exact_mixed_fnr_dist(true_numbers, p=0.5):
    num_pos, num_zero, num_neg = true_numbers

    counts = np.arange(num_pos + num_neg + 1)
    values = counts / (num_pos + num_neg)
    probabilities = binom(n=(num_pos + num_neg), p=p).pmf(counts)
    return values, probabilities


def compute_exact_pos_fnr_dist(true_numbers, p=0.5):
    num_pos, num_zero, num_neg = true_numbers

    counts = np.arange(num_pos + 1)
    values = counts / num_pos
    probabilities = binom(n=num_pos, p=p).pmf(counts)
    return values, probabilities


def compute_exact_neg_fnr_dist(true_numbers, p=0.5):
    num_pos, num_zero, num_neg = true_numbers

    counts = np.arange(num_neg + 1)
    values = counts / num_neg
    probabilities = binom(n=num_neg, p=p).pmf(counts)
    return values, probabilities


def simplify_statistic_distribution(statistic_values, probabilities):
    probabilities = probabilities.ravel()[np.argsort(statistic_values, axis=None)]
    statistic_values = np.sort(
        statistic_values, axis=None
    )  # Now they are in the same order, with statistic sorted
    # https://stackoverflow.com/a/30003565
    statistic_values, index_cutoffs = np.unique(statistic_values, return_index=True)
    # https://stackoverflow.com/a/35877759
    probabilities = np.add.reduceat(probabilities, index_cutoffs)
    return statistic_values, probabilities


# main workhorse here
### Seems to work for only relatively small n, e.g. 10 is fine
### 91 led to over 200GB of memory being used, running for hours, and still not finishing...
### Yes, b/c number of calculations is something like O((n^2/3)^3), O(n^6)
###  which for n=10 is only about 30k, but for n=91 is 21 Billion.
### OK, now I remember why I was so hesitant to make this originally and had in mind MC simulations
### of the null instead -- wow this is bad scaling. Also if I have to do MC anyway, honestly at that point
### I might just as well also do null for spearman and whoever else too, seriously at this point...
def compute_exact_null_distributions(true_numbers, p=0.5):
    num_pos, num_zero, num_neg = true_numbers

    pos_marginal_distribution = binom(n=num_pos, p=p).pmf(np.arange(num_pos + 1))
    zero_marginal_distribution = binom(n=num_zero, p=p).pmf(np.arange(num_zero + 1))
    neg_marginal_distribution = binom(n=num_neg, p=p).pmf(np.arange(num_neg + 1))
    # because they're all mutually independent
    joint_distribution = outer_product(
        [
            pos_marginal_distribution,
            zero_marginal_distribution,
            neg_marginal_distribution,
        ]
    )

    mixed_jaccard = np.zeros(joint_distribution.shape)
    pos_jaccard = np.zeros(joint_distribution.shape)
    neg_jaccard = np.zeros(joint_distribution.shape)

    mixed_fdr = np.zeros(joint_distribution.shape)
    pos_fdr = np.zeros(joint_distribution.shape)
    neg_fdr = np.zeros(joint_distribution.shape)

    for num_plus_plus, num_zero_plus, num_neg_plus in product(
        range(num_pos + 1), range(num_zero + 1), range(num_neg + 1)
    ):
        hits_pos = num_plus_plus
        false_hits_pos = num_zero_plus + num_neg_plus
        misses_pos = num_pos - hits_pos

        misses_neg = num_neg_plus
        false_hits_neg = misses_pos + (num_zero - num_zero_plus)
        hits_neg = num_neg - misses_neg

        index = (num_plus_plus, num_zero_plus, num_neg_plus)

        mixed_jaccard[index] = (hits_pos + hits_neg) / (
            num_pos + num_neg + false_hits_pos + false_hits_neg
        )
        pos_jaccard[index] = hits_pos / (num_pos + false_hits_pos)
        neg_jaccard[index] = hits_neg / (num_neg + false_hits_neg)

        mixed_fdr[index] = (false_hits_pos + false_hits_neg) / np.maximum(
            1, false_hits_pos + hits_pos + false_hits_neg + hits_neg
        )
        pos_fdr[index] = false_hits_pos / np.maximum(1, false_hits_pos + hits_pos)
        neg_fdr[index] = false_hits_neg / np.maximum(1, false_hits_neg + hits_neg)

    distributions = {}
    distributions["mix_jaccard"] = simplify_statistic_distribution(
        mixed_jaccard, joint_distribution
    )
    distributions["pos_jaccard"] = simplify_statistic_distribution(
        pos_jaccard, joint_distribution
    )
    distributions["neg_jaccard"] = simplify_statistic_distribution(
        neg_jaccard, joint_distribution
    )

    distributions["mix_fdr"] = simplify_statistic_distribution(
        mixed_fdr, joint_distribution
    )
    distributions["pos_fdr"] = simplify_statistic_distribution(
        pos_fdr, joint_distribution
    )
    distributions["neg_fdr"] = simplify_statistic_distribution(
        neg_fdr, joint_distribution
    )

    distributions["mix_fnr"] = compute_exact_mixed_fnr_dist(
        true_numbers=true_numbers, p=p
    )
    distributions["pos_fnr"] = compute_exact_pos_fnr_dist(
        true_numbers=true_numbers, p=p
    )
    distributions["neg_fnr"] = compute_exact_neg_fnr_dist(
        true_numbers=true_numbers, p=p
    )

    return distributions
