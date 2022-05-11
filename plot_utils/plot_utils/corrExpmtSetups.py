import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from string import capwords

from .corrVariants import corrPlotter
from .plot_utils import add_experiments, reverse_dict

corr_expt_mode_indices, corr_expt_mode_definitions = add_experiments(corrPlotter.mode_indices, corrPlotter.mode_definitions)

class corrExpmtSetupsPlotter(corrPlotter):
    mode_indices = corr_expt_mode_indices
    mode_definitions = corr_expt_mode_definitions
    
    mode_indices_r = reverse_dict(corr_expt_mode_indices)
    mode_definitions_r = [reverse_dict(dictionary) for dictionary in corr_expt_mode_definitions]
    
    def load_performance_results(self, metric_name):
        
        performance = np.zeros((*self.methods_shape, self.number_simulations, len(self.percentages)))
        
        for experiment_condition, control_condition in product(enumerate(["gluttonous", "picky"]), repeat=2):
            experiment_type_index, experiment_type = experiment_condition
            control_type_index, control_type = control_condition
            
            shorthand = '{}{}'.format(experiment_type[0], control_type[0])
            base_expt_setup_filename = self.base_result_filename.format(shorthand)
        
            base_filename = "{}/pct{{}}/{}".format(self.results_dir, base_expt_setup_filename)
            filenames = [base_filename.format(percentage) for percentage in self.percentages]
        
            for index, filename in enumerate(filenames):
                npzfile = np.load(filename)
                performance[experiment_type_index, control_type_index, ..., index] = npzfile[metric_name]

        return performance
    
    def plot_experiment_setup(self, experiment_type, control_type, color, ax, performance, multiindex, start_index=0):
        setup_multiindex = self.variant_to_multiindex(experiment_type=experiment_type, control_type=control_type)
        performance = performance[(*setup_multiindex, ...)]
        label = capwords('{} {}'.format(experiment_type, control_type)).replace(' ', '-')
        self.plot_metric(ax, performance[(*multiindex, ...)], color=color, label=label, start_index=start_index)
        return ax
    
    def plot_variant(self, ax, performance, corr_type, expl_type, avg_type='normalized', y_title='', start_index=0):
        multiindex = self.variant_to_multiindex(corr_type=corr_type, expl_type=expl_type, avg_type=avg_type)
        self.plot_experiment_setup('gluttonous', 'picky', 'green', ax, performance, multiindex, start_index=start_index)
        self.plot_experiment_setup('picky', 'gluttonous', 'deeppink', ax, performance, multiindex, start_index=start_index)
        self.plot_experiment_setup('picky', 'picky', 'hotpink', ax, performance, multiindex, start_index=start_index)
        self.plot_experiment_setup('gluttonous', 'gluttonous', 'darkgreen', ax, performance, multiindex, start_index=start_index)
        
        title = '{} {} (Averages of {})'.format(expl_type.capitalize(), corr_type.capitalize(), avg_type.capitalize())
        self.add_labels(ax, title=title, y_title=y_title, start_index=start_index)
        return ax
    
    def plot_all_variants(self, performance, avg_type='normalized', y_title='', start_index=0):
        nrows = len(self.get_choices("corr_type")) * len(self.get_choices("expl_type"))
        fig, axes = plt.subplots(nrows=nrows, figsize=(20, 10*nrows))
        for counter, (corr_type, expl_type) in enumerate(product(self.get_choices("corr_type"), self.get_choices("expl_type"))):
            self.plot_variant(axes[counter], performance, corr_type=corr_type, 
                              expl_type=expl_type, avg_type=avg_type, y_title=y_title, start_index=start_index)
        return axes