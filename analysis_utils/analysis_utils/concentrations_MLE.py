import re

from zipfile import BadZipfile

import numpy as np
from simulations.concentrations.MLE import get_DM_score_function, get_NB_score_function
from simulations.concentrations import get_plugin_categorical_concentration

from scipy.optimize import root_scalar, fsolve

base_relative_abundances = [1e-4, 1e-3, 1e-2]

relative_abundances = [relative_abundance * number
                       for relative_abundance 
                       in base_relative_abundances
                       for number in (1,2,5) 
                       for repeat in range(10)]

relative_abundances += [1-sum(relative_abundances)]
frequencies = np.array(relative_abundances)

def get_DM_MLE_single_batch(batch):
    max_guess_value = 1000000    
    # use frequencies as global variable -- naughty
    score_function = get_DM_score_function(batch, frequencies)
    # super inefficient b/c plugin also slow and would be better to
    # read results from a file because we already have them. but i'm lazy
    guess_value = get_plugin_categorical_concentration(batch)
    if guess_value >= max_guess_value:
        guess_value = max_guess_value/2.
    # do f(a) and f(b) have the same signs? if so then scipy won't let
    # you use bracketed method. kind of dumb, like a root might still exist
    if (score_function(1./max_guess_value) > 0) == (score_function(max_guess_value) > 0):
        result = fsolve(func=score_function, x0=guess_value, full_output=True)
        # if exit code is not 1 (successful) then usually means it failed to converge
        # after many steps because "root" is at infinity (usually, but you know sometimes 
        # there's numerical instability and yadda yadda). so if the root-finding algorithm 
        # exited unsuccessfully and the value is large, assume that's what happened.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
        if result[2] != 1 and result[0] > max_guess_value:
            result = np.inf
        else:
            result = result[0]
    else:
        result = root_scalar(f=score_function, x0=guess_value, bracket=(1./max_guess_value,max_guess_value))
        result = result.root # want result to be scalar, not `RootResults`
    return result

def get_DM_MLE(filename, varname='droplets'):

    sim_num = re.findall(r'.*\.([0-9]+)\.npz', filename)[0]
    try:
        results_file = np.load('concentration_ML_estimates/simulation.{}.compositional.npz'.format(sim_num))
        assert set(results_file.files) == {"small_batches", "medium_batches", "whole_sim"}
    except (BadZipfile, FileNotFoundError, AssertionError):
        npzfile = np.load(filename)
        droplets = npzfile[varname]
        
        try:
            assert len(frequencies) == droplets.shape[1]
        except AssertionError:
            # because the hTPMH data is formatted oppositely for historical/legacy reasons
            droplets = droplets.T

        number_droplets, number_strains = droplets.shape

        small_val_size = 10000
        small_val_iterations = number_droplets // small_val_size
        small_val_results = np.zeros(small_val_iterations)

        for iter_num in range(small_val_iterations):
            batch = droplets[iter_num*small_val_size:(iter_num+1)*small_val_size,:]
            result = get_DM_MLE_single_batch(batch)
            small_val_results[iter_num] = max(result, 0) # negative results obviously nonsensical

        med_val_size = 500000
        med_val_iterations = number_droplets // med_val_size
        med_val_results = np.zeros(med_val_iterations)

        for iter_num in range(med_val_iterations):
            batch = droplets[iter_num*med_val_size:(iter_num+1)*med_val_size,:]
            result = get_DM_MLE_single_batch(batch)
            med_val_results[iter_num] = max(result, 0)

        whole_sim_results = np.zeros(1)
        result = get_DM_MLE_single_batch(droplets)
        whole_sim_results[0] = max(result, 0)

        results = {"small_batches":small_val_results, "medium_batches":med_val_results, "whole_sim":whole_sim_results}
        np.savez_compressed('concentration_ML_estimates/simulation.{}.compositional.npz'.format(sim_num), **results)
        
    # give map/starmap something to chew on
    return 0

def get_NB_MLE_single_batch(batch, number_strains):
    max_guess_value = 10000
    
    counts = np.sum(batch, axis=1)
    number_droplets = counts.size
    
    mean_count = np.mean(counts)

    mean_centered_counts = counts - mean_count
    count_variance = np.dot(mean_centered_counts, mean_centered_counts) / number_droplets

    if count_variance <= mean_count:
        # likelihood has no maximum except at infinity
        result = np.inf
    else:
        score_function = get_NB_score_function(counts, mean_count, number_droplets, number_strains)
        guess_value = (mean_count**2) / (number_strains*(count_variance - mean_count))
        
        if (score_function(1./max_guess_value) > 0) == (score_function(max_guess_value) > 0):
            result = fsolve(func=score_function, x0=guess_value, full_output=True)
            # if exit code is not 1 (successful) then usually means it failed to converge
            # after many steps because "root" is at infinity (usually, but you know sometimes 
            # there's numerical instability and yadda yadda). so if the root-finding algorithm 
            # exited unsuccessfully and the value is large, assume that's what happened.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
            if result[2] != 1 and result[0] > max_guess_value:
                result = np.inf
            else:
                result = result[0]
        else:
            result = root_scalar(f=score_function, x0=guess_value, bracket=(1./max_guess_value,max_guess_value))
            result = result.root # want result to be scalar, not `RootResults`
    return max(result, 0.)
    
def get_NB_MLE(filename, varname='droplets'):

    sim_num = re.findall(r'.*\.([0-9]+)\.npz', filename)[0]
    try:
        results_file = np.load('concentration_ML_estimates/simulation.{}.density.npz'.format(sim_num))
        assert set(results_file.files) == {"small_batches", "medium_batches", "whole_sim"}
    except (BadZipfile, FileNotFoundError, AssertionError):
        npzfile = np.load(filename)
        droplets = npzfile[varname]
        
        try:
            assert len(frequencies) == droplets.shape[1]
        except AssertionError:
            # because the hTPMH data is formatted oppositely for historical/legacy reasons
            droplets = droplets.T

        number_droplets, number_strains = droplets.shape

        small_val_size = 10000
        small_val_iterations = number_droplets // small_val_size
        small_val_results = np.zeros(small_val_iterations)

        for iter_num in range(small_val_iterations):
            batch = droplets[iter_num*small_val_size:(iter_num+1)*small_val_size,:]
            result = get_NB_MLE_single_batch(batch, number_strains)
            small_val_results[iter_num] = max(result, 0) # negative results obviously nonsensical

        med_val_size = 500000
        med_val_iterations = number_droplets // med_val_size
        med_val_results = np.zeros(med_val_iterations)

        for iter_num in range(med_val_iterations):
            batch = droplets[iter_num*med_val_size:(iter_num+1)*med_val_size,:]
            result = get_NB_MLE_single_batch(batch, number_strains)
            med_val_results[iter_num] = max(result, 0)

        whole_sim_results = np.zeros(1)
        result = get_NB_MLE_single_batch(droplets, number_strains)
        whole_sim_results[0] = max(result, 0)

        results = {"small_batches":small_val_results, "medium_batches":med_val_results, "whole_sim":whole_sim_results}
        np.savez_compressed('concentration_ML_estimates/simulation.{}.density.npz'.format(sim_num), **results)
        
    # give map/starmap something to chew on
    return 0