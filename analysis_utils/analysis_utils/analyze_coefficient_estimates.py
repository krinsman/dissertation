import numpy as np

from collections import defaultdict
from itertools import product

from simulations.utils import (relative_error, spearman, mixed_sign_Jaccard_similarity,
                              mixed_fnr, mixed_fdr, compute_exact_null_distributions,
                              deltacon)
                              
def true_numbers_from_truth(truth):
    """here truth is a single 10 x 10 matrix"""
    num_pos = np.sum(truth > 0)
    num_nonneg = np.sum(truth >= 0)
    num_zero = num_nonneg - num_pos
    # below line definitely wouldn't work for full truth array
    num_neg = truth.size - num_nonneg
    return (num_pos, num_zero, num_neg)
    
def strictly_worse_values(value, dist_values, better=None):
    if better is None:
        raise ValueError
    if better.lower() == "higher": # .lower to make things case-insensitive
        return (dist_values < value)
    if better.lower() == "lower": # .lower to make things case-insensitive
        return (dist_values > value)
        
def get_exact_tail_probability(value, exact_dist_vals, exact_dist_probs, better=None):
    if better is None:
        raise ValueError
    values_doing_strictly_worse = strictly_worse_values(value, exact_dist_vals, better=better)
    prob_doing_strictly_worse = np.sum(exact_dist_probs[values_doing_strictly_worse])
    prob_doing_strictly_worse = min(1., prob_doing_strictly_worse) # probability greater than 1 is fltng pt. error
    return prob_doing_strictly_worse
    
def pos_part(matrix):
    return np.where(matrix > 0, matrix, 0)

def neg_part(matrix):
    return np.where(matrix < 0, -matrix, 0)
    
def get_spearmans(truth, estimate):
    return {"raw_spearman": spearman(truth, estimate),
            "mag_spearman": spearman(np.abs(truth), np.abs(estimate)),
            "pos_spearman": spearman(pos_part(truth), pos_part(estimate)),
            "neg_spearman": spearman(neg_part(truth), neg_part(estimate))}

def get_rel_errors(truth, estimate):
    return {"rel_error": relative_error(truth, estimate),
            "pos_rel_error": relative_error(pos_part(truth), pos_part(estimate)),
            "neg_rel_error": relative_error(neg_part(truth), neg_part(estimate))}

def get_deltacons(truth, estimate):
    try:
        return {"w_deltacon": deltacon(truth, estimate),
            "u_deltacon": deltacon(np.sign(truth), np.sign(estimate))}
    except np.linalg.LinAlgError as e:
        print("truth is:\n")
        print(truth)
        print("estimate is:\n")
        print(estimate)
        raise e

unweighted_jaccard = lambda *args, **kwargs: mixed_sign_Jaccard_similarity(*args, **kwargs, unweighted=True) 

def get_classification_metrics(truth, estimate, exact_null_distributions, metric_name, callback, better):
    metrics = {"mix_"+metric_name: callback(truth, estimate),
               "pos_"+metric_name: callback(truth, estimate, positive_part=True),
               "neg_"+metric_name: callback(truth, estimate, negative_part=True)}
    
    metric_tail_probabilities = {}
    for metric_type, value in metrics.items():
        metric_tail_probabilities[metric_type+'_tp'] = get_exact_tail_probability(value, *exact_null_distributions[metric_type], better=better)
        
    return {**metrics, **metric_tail_probabilities}

get_fnrs = lambda *args: get_classification_metrics(*args, metric_name='fnr', callback=mixed_fnr, better='lower')
get_fdrs = lambda *args: get_classification_metrics(*args, metric_name='fdr', callback=mixed_fdr, better='lower')
get_jaccards = lambda *args: get_classification_metrics(*args, metric_name='jaccard', callback=unweighted_jaccard, better='higher')

class Analyzer:
    def __init__(self, coefs_filepath, simulation_dir, size, seed, number_droplets, 
                 number_simulations, mode_definitions, 
                 trimmed_strains=1, **kwargs):
        self.npzfile = np.load(coefs_filepath)
        self.base_simulation_name = '{}/{}_strains.seed_{}.{}_droplets.iteration_{{}}.npz'.format(simulation_dir, size, seed, number_droplets)
        
        self.number_simulations = number_simulations
        self.mode_definitions = mode_definitions
        self.methods_shape = tuple([len(definitions) for definitions in mode_definitions])
        
        self.trimmed_strains = trimmed_strains
        self.trimmed_size = size - trimmed_strains
        
    
    def analyze(self):
        estimate_values = np.zeros((*self.methods_shape, self.trimmed_size, self.trimmed_size, self.number_simulations))
        null_array = lambda: np.zeros((*self.methods_shape, self.number_simulations))
        analysis_results = defaultdict(null_array)

        for simulation_number in range(self.number_simulations):
            truth_npzfile = np.load(self.base_simulation_name.format(simulation_number+1))
            truth = truth_npzfile['truth'][0:-self.trimmed_strains,0:-self.trimmed_strains]
            true_numbers = true_numbers_from_truth(truth)
            exact_null_distributions = compute_exact_null_distributions(true_numbers)
            pos_truth = np.where(truth > 0, truth, 0)
            neg_truth = np.where(truth < 0, -truth, 0)

            for multiindex in product(*[range(i) for i in self.methods_shape]):

                method_variant = [mode_definition[multiindex[index]] for index, mode_definition in enumerate(self.mode_definitions)]
                # e.g. 'u_pearson_bipoint'
                variant_name = '{}_{}'.format(method_variant[-1][0],'_'.join(method_variant[0:-1]))

                estimate = self.npzfile[variant_name][..., simulation_number]
                estimate_values[(*multiindex, ..., simulation_number)] = estimate
                
                for get_values in [get_spearmans, get_rel_errors, get_deltacons]:
                    values = get_values(truth, estimate)
                    for value_type, value in values.items():
                        analysis_results[value_type][(*multiindex, simulation_number)] = value
                        
                for get_values in [get_jaccards, get_fnrs, get_fdrs]:
                    values = get_values(truth, estimate, exact_null_distributions)
                    for value_type, value in values.items():
                        analysis_results[value_type][(*multiindex, simulation_number)] = value

        return (estimate_values, analysis_results)
    
    __call__ = analyze