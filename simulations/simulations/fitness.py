import numpy as np
from itertools import product
from scipy.stats import rankdata


def group_droplets(raw_counts):

    number_droplets, number_species, number_batches = raw_counts.shape

    experiments = {
        "picky": [{} for i in range(number_batches)],
        "gluttonous": [{} for i in range(number_batches)],
    }

    controls = {
        "picky": [{} for i in range(number_batches)],
        "gluttonous": [{} for i in range(number_batches)],
    }

    relevant_droplets = [{} for i in range(number_batches)]

    for batch_number in range(number_batches):
        batch = raw_counts[:, :, batch_number]
        which_are_non_zero = batch != 0
        which_are_zero = ~which_are_non_zero
        strain_counts = np.sum(which_are_non_zero, axis=1)
        one_strain_droplets = strain_counts == 1
        two_strain_droplets = strain_counts == 2
        non_zero_and_two_strain = (
            # multiplication of booleans is same as AND
            which_are_non_zero
            & two_strain_droplets[:, np.newaxis]
            # but & is slightly faster than *
        )
        not_two_strain_droplets = ~non_zero_and_two_strain

        for i in range(number_species):
            # why did I call it a row when it is actually a column?
            boolean_row = which_are_non_zero[:, i]
            i_droplet_indices = np.where(boolean_row)[0]
            relevant_droplets[batch_number][i] = i_droplet_indices
            controls["picky"][batch_number][i] = i_droplet_indices[
                one_strain_droplets[boolean_row]
            ]

            for j in range(i):
                # implies j < i, so sorted(i,j) and sorted(j,i) will be in keys
                experiments["gluttonous"][batch_number][j, i] = i_droplet_indices[
                    which_are_non_zero[:, j][boolean_row]
                ]
                experiments["picky"][batch_number][j, i] = i_droplet_indices[
                    non_zero_and_two_strain[:, j][boolean_row]
                ]
                controls["gluttonous"][batch_number][i, j] = i_droplet_indices[
                    which_are_zero[:, j][boolean_row]
                ]
            for j in range(i + 1, number_species):
                controls["gluttonous"][batch_number][i, j] = i_droplet_indices[
                    which_are_zero[:, j][boolean_row]
                ]
    return (experiments, controls, relevant_droplets)


def get_fitness_coefficients(
    raw_counts,
    all_experiments,
    all_controls,
    method="geometric",
    experiment_type="gluttonous",
    control_type="gluttonous",
    # change this default to True since that's what we use in practice
    mark_missing_as_zero=True,
):

    number_droplets, number_species, number_batches = raw_counts.shape

    if (method == "geometric") or (method == "rank"):
        # we will pre-emptively apply logarithm to all droplets later in case of "geometric"
        # x, y will be subsets of normalized rank vector in case of "rank"
        avg_fitness = lambda x, y: np.mean(x) - np.mean(y)
    if method == "arithmetic":
        avg_fitness = lambda x, y: np.log(np.mean(x)) - np.log(np.mean(y))

    results = np.zeros((number_species, number_species, number_batches))

    for batch_number in range(number_batches):
        batch = raw_counts[:, :, batch_number]
        if method == "geometric":
            batch = np.log(batch + (batch == 0))
            # could also log-transform for rank, since log is monotonic
            # but that would arguably be waste of cpu-cycles

        experiments = all_experiments[experiment_type][batch_number]
        controls = all_controls[control_type][batch_number]

        for i, j in product(range(number_species), repeat=2):

            if i == j:  # LR coefficients zero for self-interactions
                continue

            experiment_group = experiments[tuple(sorted((i, j)))]
            if control_type == "gluttonous":
                control_group = controls[
                    j, i
                ]  # we want include j, exclude i, so NOT [i,j]
            else:
                control_group = controls[j]

            # if either is empty, thus has length 0
            if (len(experiment_group) == 0) or (len(control_group) == 0):
                if mark_missing_as_zero == True:
                    continue
                else:
                    results[i, j, batch_number] = np.nan
                    continue

            strain_j_values = batch[:, j]
            if method == "rank":
                setup_values = np.concatenate([experiment_group, control_group])
                num_experiments, num_controls = (
                    experiment_group.size,
                    control_group.size,
                )
                # assert num_experiments + num_controls == setup_values.size, "Control and Experimental Groups should not overlap"
                # assert np.all(setup_values[0:num_experiments] == experiment_group)
                # assert np.all(setup_values[num_experiments:(num_experiments + num_controls)] == control_group)
                strain_j_values = strain_j_values[setup_values]
                strain_j_values = rankdata(strain_j_values, method="average") / (
                    num_experiments + num_controls
                )
                experiment_values = strain_j_values[0:num_experiments]
                control_values = strain_j_values[
                    num_experiments : (num_experiments + num_controls)
                ]
            else:
                experiment_values = strain_j_values[experiment_group]
                control_values = strain_j_values[control_group]

            results[i, j, batch_number] = avg_fitness(experiment_values, control_values)

    return results
