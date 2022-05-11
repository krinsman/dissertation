import numpy as np

from .distributions import CPM as get_initial_droplet

def glv_update_step(
    pseudocount_vectors,
    relevant_interactions,
    relevant_baserates,
    simulation,
    rng,
    noise_loc=0.0,
):
    """`noise_loc` is DEPRECATED, (imo) its only sensible possible value is 0.
    Unfortunately in earlier iterations of the project I used a value of 1, and
    so had to introduce it as a parameter in order to be able to replicate those
    earlier and slightly problematic results. Please DO NOT USE."""
    (
        cell_pseudocount_vector,
        read_pseudocount_vector,
    ) = pseudocount_vectors  # pseudocount because value not necessarily an integer

    current_total_population = np.sum(cell_pseudocount_vector)

    cell_pseudocount_vector = np.maximum(cell_pseudocount_vector, 0.0)

    difference_of_logs = discretized_log_derivative(
        cell_pseudocount_vector,
        current_total_population,
        relevant_interactions,
        relevant_baserates,
        simulation,
        rng,
        noise_loc=noise_loc,
    )
    cell_pseudocount_change = (
        np.exp(difference_of_logs) - 1.0
    ) * cell_pseudocount_vector

    cell_pseudocount_vector += cell_pseudocount_change
    # The DNA of dead cells de facto gets counted at end of experiment
    read_pseudocount_vector += np.maximum(cell_pseudocount_change, 0.0)

    return (cell_pseudocount_vector, read_pseudocount_vector)


def discretized_log_derivative(
    cell_pseudocount_vector,
    current_total_population,
    relevant_interactions,
    relevant_baserates,
    simulation,
    rng,
    noise_loc=0.0,
):
    """Pass in total population because only want it to update once a round,
    so that results don't depend on order in which species are updated."""
    standard_deviation = np.sqrt(simulation.timestep) * simulation.noise_scale
    noise = rng.normal(
        size=len(cell_pseudocount_vector), loc=noise_loc, scale=standard_deviation
    )

    interaction_terms = relevant_interactions @ cell_pseudocount_vector

    # analogous to growth rate in (logarithmic forms of) exponential or logistic differential equations
    difference_of_logs = (
        simulation.timestep * (relevant_baserates + interaction_terms) + noise
    )
    if simulation.carrying_capacity < np.inf:
        carrying_capacity_factor = 1 - (
            current_total_population / simulation.carrying_capacity
        )
        # Avoid exponential explosion in case where both are negative
        if carrying_capacity_factor < 0:
            difference_of_logs = np.abs(difference_of_logs)
            # this does nothing on positive entries, i.e. changes negative entries only

        difference_of_logs *= carrying_capacity_factor
    return difference_of_logs


def single_droplet_simulation(simulation, initial_droplet, batch, rng):

    number_updates = batch * simulation.batch_window

    relevant_indices = np.where(initial_droplet != 0)[0]

    if len(relevant_indices) == 0:
        if simulation.spikein_rate is not None:
            # 'PCR amplification biases' are also zero so whatever
            return (initial_droplet, initial_droplet, initial_droplet)
        else:
            return (initial_droplet, initial_droplet)

    relevant_interactions = simulation.glv_interaction_coefficients[
        relevant_indices, :
    ][:, relevant_indices]
    relevant_baserates = simulation.glv_baserate_coefficients[relevant_indices]

    initial_droplet = initial_droplet[relevant_indices].astype("float64")

    pseudocount_vectors = (initial_droplet, initial_droplet)

    for update in range(number_updates):
        pseudocount_vectors = glv_update_step(
            pseudocount_vectors,
            relevant_interactions,
            relevant_baserates,
            simulation,
            rng,
        )

    final_cells, final_reads = (
        np.zeros(simulation.number_species),
        np.zeros(simulation.number_species),
    )
    final_cells[relevant_indices] = pseudocount_vectors[0]
    final_reads[relevant_indices] = pseudocount_vectors[1]
    #    final_reads *= simulation.copy_numbers
    # better to add that error in manually myself imo

    if simulation.spikein_rate is not None:
        relevant_pcr_errors = get_pcr_errors(simulation, rng, len(relevant_indices))
        pcr_errors = np.zeros(simulation.number_species)
        pcr_errors[relevant_indices] = relevant_pcr_errors
        return (final_cells, final_reads, pcr_errors)
    else:
        return (final_cells, final_reads)


def get_pcr_errors(simulation, rng, number_relevant_strains):
    initial_spikein_copy_number = rng.poisson(lam=simulation.spikein_rate)
    if initial_spikein_copy_number > 0:
        # Basically change units of measurement
        rescaled_initial_spikein_number = (
            1.0 / simulation.spikein_rate
        ) * initial_spikein_copy_number
        # multiply by a log-normal
        pcr_amplified_spikein = (
            np.exp(rng.normal(scale=simulation.pcr_noise))
            * rescaled_initial_spikein_number
        )

        strain_pcr_amplification_factors = np.exp(
            rng.normal(scale=simulation.pcr_noise, size=number_relevant_strains)
        )
        # simulate normalizing by observed final number of spike-in reads
        pcr_errors = strain_pcr_amplification_factors / pcr_amplified_spikein
        return pcr_errors
    else:
        # no spike-in copies, no observations for that droplet
        return np.zeros(number_relevant_strains)
