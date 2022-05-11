from itertools import product
from collections import defaultdict

import numpy as np

class phenopathCoefficientOrganizer:
    def __init__(self, mode_definitions, size, trimmed_strains, number_simulations,
                              output_dir, **kwargs):
        self.number_simulations = number_simulations
        self.trimmed_size = size - trimmed_strains
        
        null_array = lambda : np.zeros((self.trimmed_size,self.trimmed_size,self.number_simulations))
        self.phenopath_results = defaultdict(null_array)
        
        self.mode_definitions = mode_definitions
        self.methods_shape = tuple([len(definitions) for definitions in mode_definitions])
        
        self.output_dir = output_dir
        
    def organize(self):
        for multiindex in product(*([range(i) for i in self.methods_shape][:-1])):
            mode_names = [definition[mode] for definition, mode in zip(self.mode_definitions[0:-1], multiindex)]

            for simulation_number in range(self.number_simulations):

                filename = '{}/{}/{}/iteration_{}.npz'.format(self.output_dir, *mode_names, simulation_number+1)
                npzfile = np.load(filename)

                variant_name = '_'.join(mode_names)
                self.phenopath_results['u_'+variant_name][...,simulation_number] = npzfile['uncensored_results'][0:self.trimmed_size,0:self.trimmed_size]
                self.phenopath_results['c_'+variant_name][...,simulation_number] = npzfile['censored_results'][0:self.trimmed_size,0:self.trimmed_size]

        return self.phenopath_results
    
    __call__ = organize