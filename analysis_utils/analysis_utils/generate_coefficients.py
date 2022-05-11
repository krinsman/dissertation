import numpy as np

from collections import defaultdict
from itertools import product
from pathlib import Path

from simulations.conditional_correlations import get_any_variant_conditional_correlations
from simulations.fitness import group_droplets, get_fitness_coefficients
from simulations.utils import entrywise_average

from plot_utils.corrVariants import corrPlotter
from plot_utils.fitnessVariants import fitnessPlotter

def get_averages_across_batches(array):
    u_avg = entrywise_average(*[array[..., i] for i in range(array.shape[-1])], normalize_matrices=False)
    n_avg = entrywise_average(*[array[..., i] for i in range(array.shape[-1])], normalize_matrices=True)
    return u_avg, n_avg

log_counts_to_raw_counts = lambda log_counts: np.exp(log_counts)*(log_counts != 0)

class coefficientGenerator:
    def __init__(self, output_dir, simulation_dir, size, seed, number_droplets, number_simulations, base_coefs_name, trimmed_strains=1, **kwargs):

        self.trimmed_strains = trimmed_strains
        self.number_simulations = number_simulations
        self.output_dir = output_dir
        self.base_coefs_name = base_coefs_name
        
        Path("./" + output_dir).mkdir(parents=True, exist_ok=True)
        self.base_simulation_name = '{}/{}_strains.seed_{}.{}_droplets.iteration_{{}}.npz'.format(simulation_dir, size, seed, number_droplets)

        null_array = lambda : np.zeros((size-1,size-1,number_simulations))
        self.u_avg_coefficients = defaultdict(lambda : defaultdict(null_array))
        self.n_avg_coefficients = defaultdict(lambda : defaultdict(null_array))
        
    def sanitize_simulation_results(self, filename, trimmed_strains=1):
        npzfile = np.load(filename)

        droplets_per_batch = int(npzfile['read_log_counts'].shape[0]/5)
        log_counts = np.zeros((droplets_per_batch,npzfile['read_log_counts'].shape[1],5))
        for batch in range(5):
            log_counts[...,batch] = npzfile['read_log_counts'][batch*droplets_per_batch:(batch+1)*droplets_per_batch,:]

        # doesn't work if trimmed_strains=0 I think
        log_counts = log_counts[:,0:-trimmed_strains,:] # Remove the annoying 'remainder' species
        return log_counts
        
    def load_sanitary_results(self, filename, trimmed_strains=1, log_counts=True):
        npzfile = np.load(filename)
        
        # doesn't work if trimmed_strains=0 I think
        raw_counts = npzfile['merged_reads'][:,0:-trimmed_strains,:]
        copy_numbers = npzfile['copy_numbers'][0:-trimmed_strains]
        
        # reshape for broadcasting -- sorry for again hard-coding axis-variable correspondence
        copy_numbers = copy_numbers.reshape((1,-1,1))
        
        less_raw_counts = raw_counts * copy_numbers
        if log_counts == False:
            return less_raw_counts
        else:
            log_counts = np.log(less_raw_counts + (less_raw_counts == 0))
            return log_counts
        
    def save_generated_coefs(self, **kwargs):
        for experiment_type, control_type in product(["picky", "gluttonous"], repeat=2):
            shorthand = experiment_type[0]+control_type[0]
            coefs_name = self.base_coefs_name.format(shorthand)
            experiment_setup = "{}-{}".format(experiment_type, control_type)
            coefs_filepath = '{}/{}'.format(self.output_dir, coefs_name)
            np.savez_compressed(coefs_filepath, 
                                **self.u_avg_coefficients[experiment_setup],
                                **self.n_avg_coefficients[experiment_setup],
                                **kwargs
                               )
            # These are now redundant so remove them. They were effectively progress bars
            for simulation_number in range(self.number_simulations):
                coefs_name = self.base_coefs_name.format('{}_{}'.format(shorthand, simulation_number+1))
                # missing_ok=True requires Python >= 3.8
                Path('{}/{}'.format(self.output_dir, coefs_name)).unlink(missing_ok=True)       

class fitnessGenerator(coefficientGenerator):
    
    methods = fitnessPlotter.get_choices("method")
    
    def generate(self, sanitization_needed=True, callback=None):
        
        for simulation_number in range(self.number_simulations):
            filename = self.base_simulation_name.format(simulation_number+1)
            
            if callback is not None:
                raw_counts, log_counts = callback(filename, self.trimmed_strains)
            elif sanitization_needed:
                log_counts = self.sanitize_simulation_results(filename, self.trimmed_strains)
                raw_counts = log_counts_to_raw_counts(log_counts)
            else:
                raw_counts = self.load_sanitary_results(filename, self.trimmed_strains, log_counts=False)
                log_counts = np.log(raw_counts + (raw_counts == 0))
            
            experiments, controls, _ = group_droplets(raw_counts)
            
            for experiment_type, control_type in product(["picky", "gluttonous"], repeat=2):
                
                fitness_coefficients = {}
                for method in self.methods:
                    fitness_coefficients[method] = get_fitness_coefficients(raw_counts, experiments,
                                                                           controls, method=method,
                                                                           experiment_type=experiment_type,
                                                                           control_type=control_type)
                
                # Don't really need this, but helps sanity, since this is relatively slow
                shorthand = experiment_type[0]+control_type[0]
                coefs_name = self.base_coefs_name.format('{}_{}'.format(shorthand, simulation_number+1))
                np.savez_compressed('{}/{}'.format(self.output_dir, coefs_name),
                            **fitness_coefficients)
                
                experiment_setup = "{}-{}".format(experiment_type, control_type)
                for method in self.methods:
                    (self.u_avg_coefficients[experiment_setup]['u_{}'.format(method)][...,simulation_number],
                     self.n_avg_coefficients[experiment_setup]['n_{}'.format(method)][...,simulation_number]
                    ) = get_averages_across_batches(fitness_coefficients[method])
                    
        self.save_generated_coefs(mode_definitions=fitnessPlotter.mode_definitions)
        
    __call__ = generate
                
class conditionalCorrelationGenerator(coefficientGenerator):
    
    corr_types = corrPlotter.get_choices("corr_type")
    expl_types = corrPlotter.get_choices("expl_type")
    
    def generate(self, statistic="correlation", sanitization_needed=True, callback=None):

        for simulation_number in range(self.number_simulations):
            filename = self.base_simulation_name.format(simulation_number+1)

            if callback is not None:
                log_counts = callback(filename, self.trimmed_strains)
            elif sanitization_needed:
                log_counts = self.sanitize_simulation_results(filename, self.trimmed_strains)
            else:
                log_counts = self.load_sanitary_results(filename, self.trimmed_strains, log_counts=True)

            experiments, controls, _ = group_droplets(log_counts)

            for experiment_type, control_type in product(["picky", "gluttonous"], repeat=2):    
                conditional_correlations = get_any_variant_conditional_correlations(log_counts,
                                                                               experiments,
                                                                               controls,
                                                                               experiment_type=experiment_type,
                                                                               control_type=control_type,
                                                                               statistic=statistic)
                # Don't really need this, but is helpful for "progress bar", i.e. sanity
                shorthand = experiment_type[0]+control_type[0]
                coefs_name = self.base_coefs_name.format('{}_{}'.format(shorthand, simulation_number+1))
                np.savez_compressed('{}/{}'.format(self.output_dir, coefs_name),
                            **conditional_correlations)

                experiment_setup = "{}-{}".format(experiment_type, control_type)
                for corr_type, expl_type in product(self.corr_types, self.expl_types):
                    variant_name = '{}_{}'.format(corr_type, expl_type)
                    (self.u_avg_coefficients[experiment_setup]['u_{}'.format(variant_name)][...,simulation_number],
                     self.n_avg_coefficients[experiment_setup]['n_{}'.format(variant_name)][...,simulation_number]
                    ) = get_averages_across_batches(conditional_correlations[variant_name])
                    
        self.save_generated_coefs(mode_definitions=corrPlotter.mode_definitions)
        
    __call__ = generate