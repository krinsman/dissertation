import numpy as np
import re
import json

from pathlib import Path

def get_strata_counts(filename, var_name='droplets', num_strains=91):
    # include preceding zeros and such
    simulation_number = re.findall(r'.*\.([0-9]+)\.npz', filename)[0]
    
    try:
        # yes, we could check more robustly for whether it exited correctly previously
        assert Path('number_strain_strata/number_strain_strata.{}.json'.format(simulation_number)).is_file()
        return 0
    except AssertionError:

        npzfile = np.load(filename)
        droplets = npzfile[var_name]

        try:
            assert droplets.shape[0] == num_strains
        except AssertionError:
            droplets = droplets.T
        droplets = droplets.T

        is_nonzero = (droplets != 0)
        strain_counts = np.sum(is_nonzero, axis=1)

        strata_levels = np.unique(strain_counts)

        strata_counts = {}
        for strata_level in strata_levels:
            # json module objects to 'int64' instead of converting them 'int', 
            # so we get to do the conversion ourselves... yay
            strata_counts[int(strata_level)] = int(np.sum(strain_counts == strata_level))

        Path('number_strain_strata').mkdir(parents=True, exist_ok=True)
        with open('number_strain_strata/number_strain_strata.{}.json'.format(simulation_number), 'w') as file_pointer:
            json.dump(strata_counts, file_pointer)
        return 0