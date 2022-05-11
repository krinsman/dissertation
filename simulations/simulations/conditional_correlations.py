import numpy as np
from .utils import pearson, spearman, covariance, rank_covariance
from itertools import product


def get_conditional_correlations(log_counts, statistic="correlation"):
    number_droplets, number_species, number_batches = log_counts.shape

    linear = np.zeros((number_species, number_species, number_batches))
    linear_bipoint = np.zeros(linear.shape)
    rank = np.zeros(linear.shape)
    rank_bipoint = np.zeros(linear.shape)

    if statistic == "correlation":
        linear_callback = pearson
        rank_callback = spearman
    elif statistic == "covariance":
        linear_callback = covariance
        rank_callback = rank_covariance

    for batch_number in range(number_batches):
        batch = log_counts[..., batch_number]
        which_non0 = batch != 0

        for species2 in range(number_species):
            relevant_droplets = which_non0[:, species2]
            outcome = batch[:, species2][relevant_droplets]
            for species1 in range(number_species):

                if species1 == species2:
                    continue  # see comments below
                    # this way we avoid needing to do post-processing
                    # with fill_diagonal, which I tend to forget to do
                predictor = batch[:, species1][relevant_droplets]
                linear[species1, species2, batch_number] = linear_callback(
                    predictor, outcome
                )
                rank[species1, species2, batch_number] = rank_callback(
                    predictor, outcome
                )

                predictor_bipoint = which_non0[:, species1][relevant_droplets]
                linear_bipoint[species1, species2, batch_number] = linear_callback(
                    predictor_bipoint, outcome
                )
                rank_bipoint[species1, species2, batch_number] = rank_callback(
                    predictor_bipoint, outcome
                )

    return {
        "linear_regular": linear,
        "linear_bipoint": linear_bipoint,
        "rank_regular": rank,
        "rank_bipoint": rank_bipoint,
    }


def get_any_variant_conditional_correlations(
    log_counts,
    all_experiments,
    all_controls,
    experiment_type="gluttonous",
    control_type="gluttonous",
    input_count_type="log",
    statistic="correlation",
):
    number_droplets, number_species, number_batches = log_counts.shape
    if input_count_type == "raw":
        log_counts = np.log(
            log_counts + (log_counts == 0)
        )  # log-transform non-zero values. remember log(1)=0

    linear = np.zeros((number_species, number_species, number_batches))
    linear_bipoint = np.zeros(linear.shape)
    rank = np.zeros(linear.shape)
    rank_bipoint = np.zeros(linear.shape)

    if statistic == "correlation":
        linear_callback = pearson
        rank_callback = spearman
    elif statistic == "covariance":
        linear_callback = covariance
        rank_callback = rank_covariance

    for batch_number in range(number_batches):
        batch = log_counts[..., batch_number]
        which_non0 = (batch != 0).astype("float64")

        experiments = all_experiments[experiment_type][batch_number]
        controls = all_controls[control_type][batch_number]

        for species1, species2 in product(range(number_species), repeat=2):

            if species1 == species2:
                continue  # these are 1 for regular, 0 (by convention) for bipoint,
                # convention is just to mark them all as 0 for consistency (with each other and LR coefs)
                # and also because default assumed (self-)interaction should be 0 anyway

            experiment_group = experiments[tuple(sorted((species1, species2)))]
            if control_type == "gluttonous":
                control_group = controls[
                    species2, species1
                ]  # we want to include species2, exclude species1, so NOT [species1,species2]
            else:
                control_group = controls[species2]

            relevant_droplets = np.concatenate((experiment_group, control_group))
            outcome = batch[:, species2][relevant_droplets]
            predictor_regular = batch[:, species1][relevant_droplets]
            predictor_bipoint = which_non0[:, species1][relevant_droplets]

            # pearson, spearman already handle missing data cases for us,
            # so no need for extra if block like in LR code
            linear[species1, species2, batch_number] = linear_callback(
                predictor_regular, outcome
            )
            rank[species1, species2, batch_number] = rank_callback(
                predictor_regular, outcome
            )
            linear_bipoint[species1, species2, batch_number] = linear_callback(
                predictor_bipoint, outcome
            )
            rank_bipoint[species1, species2, batch_number] = rank_callback(
                predictor_bipoint, outcome
            )

    return {
        "linear_regular": linear,
        "linear_bipoint": linear_bipoint,
        "rank_regular": rank,
        "rank_bipoint": rank_bipoint,
    }
