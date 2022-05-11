import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from string import capwords
from itertools import product

from .plot_utils import Plotter, reverse_dict

fitness_mode_indices = {"method": 0, "avg_type": 1}
fitness_mode_definitions = [
    {0: "geometric", 1: "arithmetic", 2: "rank"},
    {0: "unnormalized", 1: "normalized"},
]  

class fitnessPlotter(Plotter):
    # want these all as class attributes so can use with class method
    mode_indices = fitness_mode_indices
    mode_definitions = fitness_mode_definitions
    
    mode_indices_r = reverse_dict(fitness_mode_indices)
    mode_definitions_r = [reverse_dict(dictionary) for dictionary in fitness_mode_definitions]

class fitnessVariantsPlotter(fitnessPlotter):
    
    def load_performance_results(self, metric_name):
        base_filename = "{}/pct{{}}/{}".format(self.results_dir, self.base_result_filename)
        self.filenames = [base_filename.format(percentage) for percentage in self.percentages]
        
        performance = np.zeros((*self.methods_shape, self.number_simulations, len(self.percentages)))

        for index, filename in enumerate(self.filenames):
            npzfile = np.load(filename)
            performance[..., index] = npzfile[metric_name]

        return performance
    
    def avg_types_side_by_side(self, performance, method, y_title="", start_index=0):
        fig, ax = plt.subplots(figsize=(20, 10))

        method_code = self.variant_to_multiindex(method=method)[0]
        performance = performance[method_code, ...]
        title = '{}'.format(method.capitalize())

        colors = {"unnormalized": "red", "normalized": "green"}
        for avg_type in self.get_choices("avg_type"):
            avg_type_code = self.variant_to_multiindex(avg_type=avg_type)[0]
            color = colors[avg_type]
            self.plot_metric(ax, performance[avg_type_code,...], color=color, 
                             start_index=start_index, label=avg_type.capitalize())
        self.add_labels(ax, title=title, y_title=y_title, start_index=start_index)
        return ax

    def get_color_name(self, method):
        if method == "arithmetic":
            return "darkorange"
        elif method == "geometric":
            return "dodgerblue"
        elif method == "rank":
            return "chartreuse"
    
    def plot_variant(self, method, ax, performance, avg_type, start_index=0):        
        variant_multiindex = self.variant_to_multiindex(method=method)
        performance = performance[(*variant_multiindex, ...)]
        label = capwords('{}'.format(method.capitalize()))
        color = self.get_color_name(method)
        avg_type_code = self.variant_to_multiindex(avg_type=avg_type)[0]
        self.plot_metric(ax, performance[avg_type_code, ...], color=color, label=label, start_index=start_index)
        return ax
    
    def plot_variants(self, ax, performance, avg_type, y_title, start_index):
        for method in self.get_choices("method"):
            self.plot_variant(method, ax, performance, avg_type, start_index)
            
        title = 'Averages of {}'.format(avg_type.capitalize())
        self.add_labels(ax, title=title, y_title=y_title, start_index=start_index)

    # https://stackoverflow.com/a/37232760
    def methods_side_by_side(self,
        performance, avg_type="normalized", y_title="", start_index=0
    ):
        fig, ax = plt.subplots(nrows=1, figsize=(20, 10))
        
        self.plot_variants(ax, performance, avg_type, y_title, start_index)

        return ax