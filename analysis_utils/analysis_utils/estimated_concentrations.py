import numpy as np
import re

from zipfile import BadZipfile

from simulations.concentrations import get_plugin_categorical_concentration, get_plugin_density_concentration

def get_concentration_estimates(filename):
    
    sim_num = re.findall(r'.*\.([0-9]+)\.npz', filename)[0]
    try:
        results_file = np.load('concentration_estimates/simulation.{}.npz'.format(sim_num))
        assert set(results_file.files) == {"small_batches", "medium_batches", "whole_sim"}
    except (BadZipfile, FileNotFoundError, AssertionError):
        npzfile = np.load(filename)
        droplets = npzfile['droplets']

        number_droplets, number_strains = droplets.shape

        small_val_size = 10000
        small_val_iterations = number_droplets // small_val_size
        small_val_results = np.zeros((2, small_val_iterations))

        for iter_num in range(small_val_iterations):
            batch = droplets[iter_num*small_val_size:(iter_num+1)*small_val_size,:]
            small_val_results[0,iter_num] = get_plugin_density_concentration(batch)
            small_val_results[1,iter_num] = get_plugin_categorical_concentration(batch)

        med_val_size = 500000
        med_val_iterations = number_droplets // med_val_size
        med_val_results = np.zeros((2, med_val_iterations))

        for iter_num in range(med_val_iterations):
            batch = droplets[iter_num*med_val_size:(iter_num+1)*med_val_size,:]
            med_val_results[0,iter_num] = get_plugin_density_concentration(batch)
            med_val_results[1,iter_num] = get_plugin_categorical_concentration(batch)

        whole_sim_results = np.zeros(2)
        whole_sim_results[0] = get_plugin_density_concentration(droplets)
        whole_sim_results[1] = get_plugin_categorical_concentration(droplets)

        results = {"small_batches":small_val_results, "medium_batches":med_val_results, "whole_sim":whole_sim_results}
        np.savez_compressed('concentration_estimates/simulation.{}.npz'.format(sim_num), **results)
        
    # give map/starmap something to chew on
    return 0