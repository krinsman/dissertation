from simulations.conditional_local_chi2_cpm import get_all_conditional_pairwise_pearson_categorical_divergences_and_p_values
import time
import datetime
import numpy as np
from pathlib import Path
from zipfile import BadZipfile

import re

### Make Globals it's bad
base_relative_abundances = [1e-4, 1e-3, 1e-2]

relative_abundances = [relative_abundance * number
                       for relative_abundance 
                       in base_relative_abundances
                       for number in (1,2,5) 
                       for repeat in range(10)]

relative_abundances += [1-sum(relative_abundances)]
frequencies = np.array(relative_abundances)

rate = 2
### don't use globals it's bad

def get_conditional_pairwise_results(simulation_filename, simulation_basename, number_simulations, variable_name='droplets'):
    """Get local/pairwise results for a given simulation."""
    prettify = lambda integer: str(integer).zfill(len(str(number_simulations)))

    simulation_number = re.findall('npzfiles/{}_results\.([0-9]+)\.npz'.format(simulation_basename), simulation_filename)[0]
    
    results_filename = 'conditional_pairwise_results/{}_results.{}.npz'.format(simulation_basename, prettify(simulation_number))
    results_file = Path(results_filename)
    
    npzfile = np.load(simulation_filename)
    droplets = npzfile[variable_name]
    
    try:
        assert droplets.shape[1] == frequencies.size
    except AssertionError:
        droplets = droplets.T
        assert droplets.shape[1] == frequencies.size
    
    # simulation may have already ran successfully on previous attempt
    try:
        np.load(results_filename)
        return None
    except (BadZipfile, FileNotFoundError): # file is corrupted or does not exist
        results_file.unlink(missing_ok=True) # delete corrupted file if it exists
        start_time = time.time()
        results_dict = get_all_conditional_pairwise_pearson_categorical_divergences_and_p_values(droplets, frequencies, rate=rate, differences=True, observed=True)
        runtime = time.time() - start_time

        with open('notebook_logs/runtime.{}.log'.format(prettify(simulation_number)), 'a') as file_pointer:
            # https://stackoverflow.com/a/775095/10634604
            runtime_string = str(datetime.timedelta(seconds=runtime))
            file_pointer.write('\nRuntime for the pairwise/local hypothesis tests was {} in Hours:Minutes:Seconds.\n'.format(runtime_string))

        results_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(results_filename, **results_dict)
        return None