import numpy as np
from itertools import combinations  # iterable over combinations
from scipy.special import comb  # number combinations
from scipy.stats import chi2

### Compute a chi-squared test for a compound Poisson multinomial (cPM) distribution with a known frequency vector and rate


def get_exactly_1_or_2_strain_probs(frequency_vector, rate=2):
    frequency_vector = np.array(frequency_vector)

    exactly_2_strain_probs = get_exactly_2_strain_probs(frequency_vector, rate=rate)
    exactly_1_strain_probs = get_exactly_1_strain_probs(frequency_vector, rate=rate)
    np.fill_diagonal(exactly_2_strain_probs, exactly_1_strain_probs)

    return exactly_2_strain_probs


def get_exactly_2_strain_probs(frequency_vector, rate=2):
    """Note this is SUPER inefficient since it computes twice as many numbers as are ultimately used."""
    at_least_1_cell_probs = 1.0 - np.exp(-rate * frequency_vector)
    # outer product
    at_least_1_cell_of_each_probs = np.outer(
        at_least_1_cell_probs, at_least_1_cell_probs
    )

    # entry (i,j) is f_i + f_j
    exp_frequency_vector = np.exp(frequency_vector)
    pairwise_freq_sums = np.log(np.outer(exp_frequency_vector, exp_frequency_vector))
    # probability N^{(k)}(0) = 0 for all k not equal to i or j (maybe i or j are 0 too, maybe not)
    all_others_0_probs = np.exp(-(1.0 - pairwise_freq_sums) * rate)

    exactly_2_strain_probs = at_least_1_cell_of_each_probs * all_others_0_probs
    return np.triu(exactly_2_strain_probs, 1)


def get_exactly_1_strain_probs(frequency_vector, rate=2):
    at_least_1_cell_probs = 1.0 - np.exp(-rate * frequency_vector)
    no_other_strain_probs = np.exp(-(1.0 - frequency_vector) * rate)
    exactly_1_strain_probs = at_least_1_cell_probs * no_other_strain_probs
    return exactly_1_strain_probs


def get_observed_counts(batch):
    number_droplets, number_strains = batch.shape
    observed_1_or_2_counts = np.zeros((number_strains, number_strains)).astype(int)

    strain_is_present = batch != 0
    num_cells_in_each_droplet = np.sum(strain_is_present, axis=1)

    # not as efficient as it could be writing in C but whatever
    empty_droplet_indices = num_cells_in_each_droplet == 0
    single_strain_droplet_indices = num_cells_in_each_droplet == 1
    double_strain_droplet_indices = num_cells_in_each_droplet == 2
    multi_strain_droplet_indices = num_cells_in_each_droplet > 2

    single_strain_droplets = batch[single_strain_droplet_indices, :]
    for strain in range(number_strains):
        observed_1_or_2_counts[strain, strain] = np.sum(
            single_strain_droplets[:, strain] != 0
        )

    double_strain_droplets = batch[double_strain_droplet_indices, :]
    for strain1, strain2 in combinations(range(number_strains), 2):
        # again not as efficient as it could be in computing same thing multiple times
        strain1_droplet_indices = double_strain_droplets[:, strain1] != 0
        strain2_droplet_indices = double_strain_droplets[:, strain2] != 0
        both_strain_droplet_indices = strain1_droplet_indices & strain2_droplet_indices
        observed_1_or_2_counts[strain1, strain2] = np.sum(both_strain_droplet_indices)

    num_empty_droplets = np.sum(empty_droplet_indices)
    num_multi_strain_droplets = (
        number_droplets - num_empty_droplets - np.sum(observed_1_or_2_counts)
    )
    # again, not efficient, unnecessary computations, etc
    assert num_multi_strain_droplets == np.sum(multi_strain_droplet_indices)
    return num_empty_droplets, observed_1_or_2_counts, num_multi_strain_droplets


def get_expected_probs(frequency_vector, rate=2):
    prob_empty_droplet = np.exp(-rate)
    exactly_1_or_2_strain_probs = get_exactly_1_or_2_strain_probs(
        frequency_vector, rate=rate
    )
    prob_multi_strain_droplet = (
        1.0 - np.sum(exactly_1_or_2_strain_probs) - prob_empty_droplet
    )
    return (
        prob_empty_droplet,
        exactly_1_or_2_strain_probs,
        prob_multi_strain_droplet,
    )


def get_expected_counts(number_droplets, frequency_vector, rate=2):
    (
        prob_empty_droplet,
        exactly_1_or_2_strain_probs,
        prob_multi_strain_droplet,
    ) = get_expected_probs(frequency_vector, rate=rate)
    return (
        number_droplets * prob_empty_droplet,
        number_droplets * exactly_1_or_2_strain_probs,
        number_droplets * prob_multi_strain_droplet,
    )


def get_differences_from_expected(batch, frequency_vector, rate=2):
    number_droplets, number_strains = batch.shape

    (
        observed_num_empty_droplets,
        observed_1_or_2_counts,
        observed_num_multi_strain_droplets,
    ) = get_observed_counts(batch)
    (
        expected_num_empty_droplets,
        expected_1_or_2_counts,
        expected_num_multi_strain_droplets,
    ) = get_expected_counts(number_droplets, frequency_vector, rate=rate)

    differences = (observed_num_empty_droplets - expected_num_empty_droplets,
        observed_1_or_2_counts - expected_1_or_2_counts,
        observed_num_multi_strain_droplets - expected_num_multi_strain_droplets)
    
    return differences

    
def get_chi_squared_statistic(batch, frequency_vector, rate=2):
    number_droplets, number_strains = batch.shape

    (
        observed_num_empty_droplets,
        observed_1_or_2_counts,
        observed_num_multi_strain_droplets,
    ) = get_observed_counts(batch)
    (
        expected_num_empty_droplets,
        expected_1_or_2_counts,
        expected_num_multi_strain_droplets,
    ) = get_expected_counts(number_droplets, frequency_vector, rate=rate)

    dof = (1 + comb(number_strains, 2, exact=True) + number_strains + 1) - 1
    # very inefficient/unnecessary check
    assert dof == np.sum(expected_1_or_2_counts != 0) + 1  # 2 - 1 = 1
    # divide lower triangle entries by 1. instead of 0., to avoid error from dividing 0 by 0
    exactly_1_or_2_strain_chi2 = (
        observed_1_or_2_counts - expected_1_or_2_counts
    ) ** 2 / (expected_1_or_2_counts + (expected_1_or_2_counts == 0))
    chi_squared_statistic = np.sum(exactly_1_or_2_strain_chi2)
    chi_squared_statistic += (
        (observed_num_empty_droplets - expected_num_empty_droplets) ** 2
    ) / expected_num_empty_droplets
    chi_squared_statistic += (
        (observed_num_multi_strain_droplets - expected_num_multi_strain_droplets) ** 2
    ) / expected_num_multi_strain_droplets
    percentile = chi2.sf(x=chi_squared_statistic, df=dof)
        
    return chi_squared_statistic, percentile


from statsmodels.distributions.empirical_distribution import ECDF

# NOTE: Monte Carlo p-values are result of applying SURVIVAL function, not CDF
# i.e. it's 1 - CDF value
def get_monte_carlo_ecdf(
    number_droplets,
    frequency_vector,
    rate=2,
    monte_carlo_seed=42,
    monte_carlo_trials=10000,
):

    probs_to_flatten = get_expected_probs(frequency_vector, rate=rate)
    probs = np.array(
        [probs_to_flatten[0]]
        + list(probs_to_flatten[1].ravel()[np.flatnonzero(probs_to_flatten[1])])
        + [probs_to_flatten[2]]
    )

    rng = np.random.default_rng(monte_carlo_seed)
    multinomial_trials = rng.multinomial(
        n=number_droplets, pvals=probs, size=monte_carlo_trials
    )

    expected_counts = number_droplets * probs

    monte_carlo_chi2_stats = np.sum(
        ((multinomial_trials - expected_counts) ** 2 / expected_counts), axis=1
    )

    ecdf = ECDF(monte_carlo_chi2_stats)

    return ecdf
