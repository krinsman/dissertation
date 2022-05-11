import numpy as np

from .simulation import get_initial_droplet, single_droplet_simulation
from .fitness import group_droplets, get_fitness_coefficients

from multiprocessing import cpu_count, Pool, get_context

from pathlib import Path


class dropletSimulation:
    def __init__(
        self,
        number_species,
        number_droplets,
        number_batches,
        copy_numbers,
        frequency_vector,
        glv_interaction_coefficients,
        glv_baserate_coefficients,
        batch_window=200,
        poisson_rate=2,
        timestep=0.0001,
        noise_scale=8,
        carrying_capacity=268, # default used to be 10,000 but Fangchao said 200-300 is a more realistic number, 268 result of calculation
        merging_error=0.0,
        seed=42,
        spikein_rate=32,
        pcr_noise=0.1,
    ):

        assert isinstance(
            number_species, int
        ), "The number of species should be an integer."
        assert isinstance(
            number_droplets, int
        ), "The number of droplets per batch should be an integer."
        assert isinstance(
            number_batches, int
        ), "The number of batches should be an integer."
        assert isinstance(
            batch_window, int
        ), "The batch window should be an integer corresponding to the number of simulation timesteps that should occur between each batch."

        def validate_float_or_int(quantity, quantity_name):
            if isinstance(quantity, int):
                quantity = float(quantity)
            float_or_int_type_message = (
                "The {} should be represented by an integer or a float."
            )
            assert isinstance(quantity, float), float_or_int_type_message.format(
                quantity_name
            )
            return quantity

        poisson_rate = validate_float_or_int(
            poisson_rate, "rate for the Poisson distribution"
        )
        timestep = validate_float_or_int(
            timestep, "length of the timestep for the simulation"
        )
        noise_scale = validate_float_or_int(
            noise_scale, "scale of the random noise generated for the simulation"
        )
        carrying_capacity = validate_float_or_int(
            carrying_capacity, "carrying capacity of each droplet"
        )

        merging_error = validate_float_or_int(
            merging_error, "portion of droplets merged after end of simulation"
        )
        assert (
            0 <= merging_error <= 1
        ), "Merging error refers to portion of droplets merged after end of simulation, should be in between 0 and 1."

        if pcr_noise is not None:
            assert (
                spikein_rate is not None
            ), "No default value for expected number of spike-in gene copies added to PCR mix, so this needs to be specified in addition to `pcr_noise` to simulate PCR amplification errors."

        if spikein_rate is not None:
            spikein_rate = validate_float_or_int(
                spikein_rate,
                "expected number of the Poisson-distributed copies of spike-in gene added to PCR mix.",
            )
            if pcr_noise is None:
                pcr_noise = 1.0  # Default log-sd for log-normal
            pcr_noise = validate_float_or_int(
                pcr_noise,
                "standard deviation of logarithm of distribution corresponding to (rescaled) PCR amplification factor.",
            )

        copy_numbers_name = "copy numbers"
        frequency_vector_name = "species frequencies"
        glv_baserate_coefficients_name = (
            "base rate coefficients for the generalized Lotka-Volterra equations"
        )
        glv_interaction_coefficients_name = (
            "interaction coefficients for the generalized Lotka-Volterra equations"
        )

        vector_type_message = (
            "The vector of {} should be represented as a 1D NumPy array."
        )
        assert isinstance(copy_numbers, np.ndarray), vector_type_message.format(
            copy_numbers_name
        )
        assert isinstance(frequency_vector, np.ndarray), vector_type_message.format(
            frequency_vector_name
        )
        assert isinstance(
            glv_baserate_coefficients, np.ndarray
        ), vector_type_message.format(glv_baserate_coefficients_name)
        assert isinstance(
            glv_interaction_coefficients, np.ndarray
        ), "The {} should be represented as a 2D NumPy array.".format(
            glv_interaction_coefficients_name
        )

        vector_shape_message = "The vector of {} should be a 1D NumPy array with number of entries equal to the number of species."
        assert copy_numbers.shape == (number_species,), vector_shape_message.format(
            copy_numbers_name
        )
        assert frequency_vector.shape == (number_species,), vector_shape_message.format(
            frequency_vector_name
        )
        assert glv_baserate_coefficients.shape == (
            number_species,
        ), vector_shape_message.format(glv_baserate_coefficients_name)
        assert glv_interaction_coefficients.shape == (
            number_species,
            number_species,
        ), "The {} should be an S x S matrix, where S denotes the number of species, represented as a 2D NumPy array.".format(
            glv_interaction_coefficients_name
        )

        self.number_species = number_species
        self.number_droplets = number_droplets
        self.number_batches = number_batches
        self.copy_numbers = copy_numbers
        self.frequency_vector = frequency_vector
        self.glv_interaction_coefficients = glv_interaction_coefficients
        self.glv_baserate_coefficients = glv_baserate_coefficients
        self.batch_window = batch_window
        self.poisson_rate = poisson_rate
        self.timestep = timestep
        self.noise_scale = noise_scale
        self.carrying_capacity = carrying_capacity
        self.merging_error = merging_error
        self.spikein_rate = spikein_rate
        self.pcr_noise = pcr_noise

        try:
            assert isinstance(seed, int)
            self.seed = np.random.SeedSequence(seed)
        except AssertionError:
            assert isinstance(
                seed, np.random.SeedSequence
            ), "The seed needs to be an instance of numpy.random.SeedSequence, either implicitly specified by an integer (corresponding to its 'entropy'), or given directly."
            self.seed = seed

    def run_simulation(
        self,
        large_batches=False,
        number_processes=None,
        maxtasksperchild=None,
        chunksize=None,
        results_dir_name=None,
    ):
        
        if large_batches == False:
            cell_pseudocount_results = np.zeros(
                (self.number_droplets, self.number_species, self.number_batches)
            )
            read_pseudocount_results = np.zeros(
                (self.number_droplets, self.number_species, self.number_batches)
            )
            if self.spikein_rate is not None:
                pcr_errors = np.zeros(
                    (self.number_droplets, self.number_species, self.number_batches)
                )
            if number_processes is None:
                number_processes = cpu_count()

            for batch_number in np.arange(self.number_batches) + 1:

                droplet_seeds = self.seed.spawn(self.number_droplets)
                droplet_rngs = [np.random.default_rng(seed) for seed in droplet_seeds]

                # Setting context to be fork is necessary for Python >= 3.8 on MacOS;
                # other Unix and Python <= 3.7 on MacOS use 'fork' by default
                # 'fork' not supported on Windows
                
                # https://stackoverflow.com/questions/49429368/how-to-solve-memory-issues-problems-while-multiprocessing-using-pool-map#comment86212360_49502189
                with get_context("fork").Pool(
                    processes=number_processes, maxtasksperchild=maxtasksperchild
                ) as final_droplet_pool:
                
                    with get_context("fork").Pool(
                        processes=number_processes, maxtasksperchild=maxtasksperchild
                    ) as initial_droplet_pool:

                        parallel_input = list(
                            zip(
                                [self.poisson_rate] * self.number_droplets,
                                [self.frequency_vector] * self.number_droplets,
                                droplet_rngs,
                            )
                        )
                        # getting initial droplets is just numpy RNGs and so really can't benefit from small chunksize I think
                        initial_droplets = initial_droplet_pool.starmap(
                            get_initial_droplet, parallel_input
                        )
                        initial_droplet_pool.close()

                    parallel_input = list(
                        zip(
                            [self] * self.number_droplets,
                            initial_droplets,
                            [batch_number] * self.number_droplets,
                            droplet_rngs,
                        )
                    )
                    # here we pass chunksize though
                    simulation_results = final_droplet_pool.starmap(
                        single_droplet_simulation, parallel_input, chunksize=chunksize
                    )
                    final_droplet_pool.close()

                if self.spikein_rate is not None:
                    (
                        cell_pseudocount_results[:, :, batch_number - 1],
                        read_pseudocount_results[:, :, batch_number - 1],
                        pcr_errors[:, :, batch_number - 1],
                    ) = zip(*simulation_results)
                else:
                    (
                        cell_pseudocount_results[:, :, batch_number - 1],
                        read_pseudocount_results[:, :, batch_number - 1],
                    ) = zip(*simulation_results)

            self.cells = dropletCounts(self, cell_pseudocount_results)
            self.reads = dropletCounts(self, read_pseudocount_results)
            if self.spikein_rate is not None:
                self.pcr_errors = dropletCounts(self, pcr_errors)

            # it's more useful as a worklow to subsequently call this function
            # manually, so I can separate effects of relic DNA and of merging
            # self.create_merging_errors()

        else:

            if results_dir_name is not None:
                Path("./" + results_dir_name).mkdir(parents=True, exist_ok=True)
                base_simulation_filename = (
                    results_dir_name
                    + "/{{}}.{}_strains.seed_{}.batch_{{}}_of_{}_batches.{}_droplets.npz"
                )
            else:
                base_simulation_filename = (
                    "/{{}}.{}_strains.seed_{}.batch_{{}}_of_{}_batches.{}_droplets.npz"
                )

            base_simulation_filename = base_simulation_filename.format(
                self.number_species,
                self.seed.entropy,
                self.number_batches,
                self.number_droplets,
            )

            # Way too much copy-pasted code -- not cool
            # Will probably have to separate this out into a function at some point
            if number_processes is None:
                number_processes = cpu_count()

            for batch_number in np.arange(self.number_batches) + 1:

                droplet_seeds = self.seed.spawn(self.number_droplets)
                droplet_rngs = [np.random.default_rng(seed) for seed in droplet_seeds]

                initial_droplet_pool = Pool(
                    processes=number_processes, maxtasksperchild=maxtasksperchild
                )
                # https://stackoverflow.com/questions/49429368/how-to-solve-memory-issues-problems-while-multiprocessing-using-pool-map#comment86212360_49502189
                final_droplet_pool = Pool(
                    processes=number_processes, maxtasksperchild=maxtasksperchild
                )
                parallel_input = list(
                    zip(
                        [self.poisson_rate] * self.number_droplets,
                        [self.frequency_vector] * self.number_droplets,
                        droplet_rngs,
                    )
                )
                # getting initial droplets is just numpy RNGs and so really can't benefit from small chunksize I think
                initial_droplets = initial_droplet_pool.starmap(
                    get_initial_droplet, parallel_input
                )
                initial_droplet_pool.close()

                parallel_input = list(
                    zip(
                        [self] * self.number_droplets,
                        initial_droplets,
                        [batch_number] * self.number_droplets,
                        droplet_rngs,
                    )
                )
                # here we pass chunksize though
                simulation_results = final_droplet_pool.starmap(
                    single_droplet_simulation, parallel_input, chunksize=chunksize
                )
                final_droplet_pool.close()

                if self.spikein_rate is not None:
                    (
                        cell_pseudocount_results,
                        read_pseudocount_results,
                        pcr_errors,
                    ) = zip(*simulation_results)
                else:
                    (
                        cell_pseudocount_results,
                        read_pseudocount_results,
                    ) = zip(*simulation_results)

                np.savez_compressed(
                    base_simulation_filename.format("true_cell_results", batch_number),
                    cell_pseudocount_results,
                )
                # doesn't free memory immediately, but hopefully gives garbage-collection a hint-hint nudge-nudge
                del cell_pseudocount_results

                np.savez_compressed(
                    base_simulation_filename.format("raw_read_results", batch_number),
                    read_pseudocount_results,
                )
                del read_pseudocount_results

                if self.spikein_rate is not None:
                    np.savez_compressed(
                        base_simulation_filename.format(
                            "pcr_error_results", batch_number
                        ),
                        pcr_errors,
                    )
                    del pcr_errors

    def create_merging_errors(self, counts):
        if self.merging_error == 0:
            print("Class instance merging error is set to 0%, so nothing happens.")
            return counts
        else:
            number_droplets_to_merge = int(
                np.floor((self.merging_error / 2) * self.number_droplets)
            )

            merging_rng = np.random.default_rng(self.seed)

            droplet_merge_indices = merging_rng.choice(
                self.number_droplets, 2 * number_droplets_to_merge, replace=False
            )

            # first half will be the droplets that get droplets merged into them
            self.droplets_merged_into_indices = droplet_merge_indices[
                number_droplets_to_merge:
            ]
            # second half are the droplets that get merged
            self.droplets_merged_indices = droplet_merge_indices[
                :number_droplets_to_merge
            ]
            # make them object attributes so can be looked into later, i.e. facilitate possible analysis

            # droplets merged into have their counts incremented by the droplets that are merged
            counts[self.droplets_merged_into_indices, ...] += counts[
                self.droplets_merged_indices, ...
            ]
            # which axis (the first) corresponds to droplets is hard-coded here, sorry
            counts = np.delete(counts, self.droplets_merged_indices, axis=0)
            return counts

    def group_droplets(self):
        self.cells.group_droplets()
        self.reads.group_droplets()

    def __getattr__(self, name):
        if (name == "reads") or (name == "cells"):
            raise AttributeError(
                "Please first use the `run_simulation` method in order to access the `{}` attribute".format(
                    name
                )
            )
        else:
            return self.__getattribute__(name)


class dropletCounts:
    def __init__(self, simulation, counts):
        assert isinstance(simulation, dropletSimulation)
        self.simulation = simulation
        assert isinstance(counts, np.ndarray)
        assert counts.shape == (
            simulation.number_droplets,
            simulation.number_species,
            simulation.number_batches,
        )
        self.counts = counts

    def __getattr__(self, name):
        if (name == "experiments") or (name == "controls"):
            raise AttributeError(
                "Please first use the `group_droplets` method in order to access the `{}` attribute".format(
                    name
                )
            )
        try:
            # get stuff like number_batches, number_droplets
            return getattr(self.simulation, name)
        except AttributeError:
            try:
                # get stuff like self.counts.shape, again for being lazy
                return getattr(self.counts, name)
            except AttributeError:
                return self.__getattribute__(name)

    # Allows us to be lazy and access droplet counts directly using square brackets
    # no need to add extra `.counts`
    def __getitem__(self, name):
        return self.counts.__getitem__(name)

    def group_droplets(self):
        self.experiments, self.controls, self.relevant_droplets = group_droplets(
            self.counts
        )

    def get_fitness_coefficients(
        self,
        method="geometric",
        experiment_type="gluttonous",
        control_type="gluttonous",
        mark_missing_as_zero=False,
    ):
        return get_fitness_coefficients(
            self.counts,
            self.experiments,
            self.controls,
            method=method,
            experiment_type=experiment_type,
            control_type=control_type,
            mark_missing_as_zero=mark_missing_as_zero,
        )

    def get_glv_coefficients(self):
        result = np.zeros((self.number_species + 1, self.number_species))

        for species in range(self.number_species):

            squiggle_B = [
                batch_number
                for counter, batch_number in enumerate(range(self.number_batches - 1))
                if (len(self.relevant_droplets[batch_number][species]) > 0)
                and (len(self.relevant_droplets[batch_number + 1][species]) > 0)
            ]
            squiggle_B = {
                counter: batch_number for counter, batch_number in enumerate(squiggle_B)
            }

            # not enough data to make any estimates
            if len(squiggle_B) == 0:
                result[:, species] = np.nan
                continue

            avg_counts = np.zeros((self.number_species + 1, len(squiggle_B)))

            for index, batch_number in squiggle_B.items():
                indices_of_relevant_droplets = self.relevant_droplets[batch_number][
                    species
                ]
                avg_counts[: self.number_species, index] = np.mean(
                    self.counts[indices_of_relevant_droplets, :, batch_number], axis=0
                )
            # all one rows for the "regression" intercept coefficients, i.e. the baserate estimate
            avg_counts[self.number_species, :] = np.ones(len(squiggle_B))

            log_diffs = np.zeros(len(squiggle_B))

            for index, batch_number in squiggle_B.items():
                latter_indices_of_relevant_droplets = self.relevant_droplets[
                    batch_number + 1
                ][species]
                log_diffs[index] = np.mean(
                    np.log(
                        self.counts[:, species, batch_number + 1][
                            latter_indices_of_relevant_droplets
                        ]
                    )
                )

                former_indices_of_relevant_droplets = self.relevant_droplets[
                    batch_number
                ][species]
                log_diffs[index] -= np.mean(
                    np.log(
                        self.counts[:, species, batch_number][
                            former_indices_of_relevant_droplets
                        ]
                    )
                )

            result[:, species] = np.linalg.pinv(avg_counts.T) @ log_diffs

        return {
            "interaction_estimates": result[: self.number_species, :],
            "baserate_estimates": result[self.number_species, :],
        }
