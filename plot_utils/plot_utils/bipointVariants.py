import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from string import capwords
from itertools import product

from .plot_utils import Plotter, reverse_dict
from .corrVariants import corrPlotter
from .fitnessVariants import fitnessPlotter

bipoint_mode_indices = {"statistic": 0, "method": 1, "avg_type": 2}
bipoint_mode_definitions = [
    {0: "fitness", 1: "correlation"},
    {0: "linear", 1: "rank"},
    {0: "unnormalized", 1: "normalized"},
] 

class bipointPlotter(Plotter):
    mode_indices = bipoint_mode_indices
    mode_definitions = bipoint_mode_definitions
    mode_indices_r = reverse_dict(bipoint_mode_indices)
    mode_definitions_r = [reverse_dict(dictionary) for dictionary in bipoint_mode_definitions]

class bipointVariantsPlotter(bipointPlotter):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # be nasty, and duck-type base_result_filename as dicts
        self.fitness_base_result_filename = self.base_result_filename["fitness"]
        self.corr_base_result_filename = self.base_result_filename["correlation"]
        
    def load_performance_results(self, metric_name):
        fitness_base_filename = "{}/pct{{}}/{}".format(self.results_dir, self.fitness_base_result_filename)
        self.fitness_filenames = [fitness_base_filename.format(percentage) for percentage in self.percentages]
        
        corr_base_filename = "{}/pct{{}}/{}".format(self.results_dir, self.corr_base_result_filename)
        self.corr_filenames = [corr_base_filename.format(percentage) for percentage in self.percentages]
        
        performance = np.zeros((*self.methods_shape, self.number_simulations, len(self.percentages)))

        linear_coef_code = self.variant_to_multiindex(method="linear")[0]
        rank_coef_code = self.variant_to_multiindex(method="rank")[0]
        
        # geometric fitness coefficient a special case of a linear correlation
        linear_fitness_code = fitnessPlotter.variant_to_multiindex(method="geometric")[0]
        rank_fitness_code = fitnessPlotter.variant_to_multiindex(method="rank")[0]
        fitness_coef_code = self.variant_to_multiindex(statistic="fitness")[0]
        for index, filename in enumerate(self.fitness_filenames):
            npzfile = np.load(filename)
            local_performance = npzfile[metric_name]
            # still making assumptions about order of modes but whatever for now
            geometric_local_performance = local_performance[linear_fitness_code, ...]
            performance[fitness_coef_code, linear_coef_code, ..., index] = geometric_local_performance
            
            rank_local_performance = local_performance[rank_fitness_code, ...]
            performance[fitness_coef_code, rank_coef_code, ..., index] = rank_local_performance
            
        linear_corr_code = corrPlotter.variant_to_multiindex(corr_type="linear")[0]
        rank_corr_code = corrPlotter.variant_to_multiindex(corr_type="rank")[0]
        bipoint_corr_code = corrPlotter.variant_to_multiindex(expl_type="bipoint")[0]
        corr_coef_code = self.variant_to_multiindex(statistic="correlation")[0]
        for index, filename in enumerate(self.corr_filenames):
            npzfile = np.load(filename)
            local_performance = npzfile[metric_name]
            # still making assumptions about order of modes but whatever for now
            linear_local_performance = local_performance[linear_corr_code, bipoint_corr_code, ...]
            performance[corr_coef_code, linear_coef_code, ..., index] = linear_local_performance
            
            rank_local_performance = local_performance[rank_corr_code, bipoint_corr_code, ...]
            performance[corr_coef_code, rank_coef_code, ..., index] = rank_local_performance
        return performance

    def avg_types_side_by_side(self, performance, statistic, method, y_title="", start_index=0):
        fig, ax = plt.subplots(figsize=(20, 10))

        multiindex = self.variant_to_multiindex(statistic=statistic, method=method)
        performance = performance[(*multiindex, ...)]
        title = '{} {}'.format(method.capitalize(), statistic.capitalize())

        colors = {"unnormalized": "red", "normalized": "green"}
        for avg_type in self.get_choices("avg_type"):
            avg_type_code = self.variant_to_multiindex(avg_type=avg_type)[0]
            color = colors[avg_type]
            self.plot_metric(ax, performance[avg_type_code,...], color=color, 
                             start_index=start_index, label=avg_type.capitalize())
        self.add_labels(ax, title=title, y_title=y_title, start_index=start_index)
        return ax

    def get_color_name(self, statistic, method):
        if statistic == "fitness":    
            if method == "linear":
                return "darkorange"
            elif method == "rank":
                return "chartreuse"            
        elif statistic == "correlation":
            if method == "linear":
                return "firebrick"
            elif method == "rank":
                return "dodgerblue"
    
    def plot_variant(self, statistic, method, ax, performance, avg_type, start_index=0):        
        variant_multiindex = self.variant_to_multiindex(statistic=statistic, method=method)
        performance = performance[(*variant_multiindex, ...)]
        label = capwords('{} {}'.format(method, statistic)).replace(' ', '-')
        color = self.get_color_name(statistic, method)
        avg_type_code = self.variant_to_multiindex(avg_type=avg_type)[0]
        self.plot_metric(ax, performance[avg_type_code, ...], color=color, label=label, start_index=start_index)
        return ax
    
    def plot_variants(self, ax, performance, avg_type, y_title, start_index, fixed_statistic="", fixed_method=""):
        assert (not fixed_statistic) or (not fixed_method)
        for statistic, method in product(self.get_choices("statistic"), self.get_choices("method")):
            is_statistic_match = (not fixed_statistic) or (statistic == fixed_statistic)
            is_method_match = (not fixed_method) or (method == fixed_method)
            if is_method_match and is_statistic_match:
                self.plot_variant(statistic, method, ax, performance, avg_type, start_index)
            
        title = 'Averages of {}'.format(avg_type.capitalize())
        if fixed_statistic:
            title = '{} - {} Only'.format(title, fixed_statistic.capitalize())
        if fixed_method:
            title = '{} - {} Only'.format(title, fixed_method.capitalize())
        self.add_labels(ax, title=title, y_title=y_title, start_index=start_index)
        

    # https://stackoverflow.com/a/37232760
    def statistic_and_method_types_side_by_side(self,
        performance, avg_type="normalized", y_title="", start_index=0
    ):
        nrows = 1 + len(self.get_choices("statistic")) + len(self.get_choices("method"))
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=nrows, figsize=(20, 10*nrows))
        
        self.plot_variants(ax1, performance, avg_type, y_title, start_index)
        self.plot_variants(ax2, performance, avg_type, y_title, start_index, fixed_statistic="fitness")
        self.plot_variants(ax3, performance, avg_type, y_title, start_index, fixed_statistic="correlation")
        self.plot_variants(ax4, performance, avg_type, y_title, start_index, fixed_method="linear")
        self.plot_variants(ax5, performance, avg_type, y_title, start_index, fixed_method="rank")

        return (ax1, ax2, ax3, ax4, ax5)