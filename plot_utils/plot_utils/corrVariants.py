import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from string import capwords
from itertools import product

from .plot_utils import Plotter, reverse_dict

corr_mode_indices = {"corr_type": 0, "expl_type": 1, "avg_type": 2}
corr_mode_definitions = [
    {0: "linear", 1: "rank"},
    {0: "regular", 1: "bipoint"},  # explanatory variable type
    {0: "unnormalized", 1: "normalized"},
]  # average type    

        
class corrPlotter(Plotter):
    mode_indices = corr_mode_indices
    mode_definitions = corr_mode_definitions
    mode_indices_r = reverse_dict(corr_mode_indices)
    mode_definitions_r = [reverse_dict(dictionary) for dictionary in corr_mode_definitions]

class corrVariantsPlotter(corrPlotter):
        
    def load_performance_results(self, metric_name):
        base_filename = "{}/pct{{}}/{}".format(self.results_dir, self.base_result_filename)
        self.filenames = [base_filename.format(percentage) for percentage in self.percentages]
        
        performance = np.zeros((*self.methods_shape, self.number_simulations, len(self.percentages)))

        for index, filename in enumerate(self.filenames):
            npzfile = np.load(filename)
            performance[..., index] = npzfile[metric_name]

        return performance
    
    def avg_types_side_by_side(self, performance, corr_type, expl_type, y_title="", start_index=0):
        fig, ax = plt.subplots(figsize=(20, 10))

        multiindex = self.variant_to_multiindex(corr_type=corr_type, expl_type=expl_type)
        performance = performance[(*multiindex, ...)]
        title = '{} {}'.format(expl_type.capitalize(), corr_type.capitalize())

        colors = {"unnormalized": "red", "normalized": "green"}
        for avg_type in self.get_choices("avg_type"):
            avg_type_code = self.variant_to_multiindex(avg_type=avg_type)[0]
            color = colors[avg_type]
            self.plot_metric(ax, performance[avg_type_code,...], color=color, 
                             start_index=start_index, label=avg_type.capitalize())
        self.add_labels(ax, title=title, y_title=y_title, start_index=start_index)
        return ax

    def get_color_name(self, corr_type, expl_type):
        if corr_type == "linear":
            if expl_type == "regular":
                return "yellow"
            elif expl_type == "bipoint":
                return "darkorange"
        elif corr_type == "rank":
            if expl_type == "regular":
                return "indigo"
            elif expl_type == "bipoint":
                return "dodgerblue"
    
    def plot_variant(self, corr_type, expl_type, ax, performance, avg_type, start_index=0):        
        variant_multiindex = self.variant_to_multiindex(corr_type=corr_type, expl_type=expl_type)
        performance = performance[(*variant_multiindex, ...)]
        label = capwords('{} {}'.format(corr_type, expl_type)).replace(' ', '-')
        color = self.get_color_name(corr_type, expl_type)
        avg_type_code = self.variant_to_multiindex(avg_type=avg_type)[0]
        self.plot_metric(ax, performance[avg_type_code, ...], color=color, label=label, start_index=start_index)
        return ax
    
    def plot_variants(self, ax, performance, avg_type, y_title, start_index, fixed_corr_type="", fixed_expl_type=""):
        assert (not fixed_corr_type) or (not fixed_expl_type)
        for corr_type, expl_type in product(self.get_choices("corr_type"), self.get_choices("expl_type")):
            is_corr_type_match = (not fixed_corr_type) or (corr_type == fixed_corr_type)
            is_expl_type_match = (not fixed_expl_type) or (expl_type == fixed_expl_type)
            if is_corr_type_match and is_expl_type_match:
                self.plot_variant(corr_type, expl_type, ax, performance, avg_type, start_index)
            
        title = 'Averages of {}'.format(avg_type.capitalize())
        if fixed_corr_type:
            title = '{} - {} Only'.format(title, fixed_corr_type.capitalize())
        if fixed_expl_type:
            title = '{} - {} Only'.format(title, fixed_expl_type.capitalize())
        self.add_labels(ax, title=title, y_title=y_title, start_index=start_index)
        

    # https://stackoverflow.com/a/37232760
    def corr_and_expl_types_side_by_side(self,
        performance, avg_type="normalized", y_title="", start_index=0
    ):
        nrows = 1 + len(self.get_choices("corr_type")) + len(self.get_choices("expl_type"))
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=nrows, figsize=(20, 10*nrows))
        
        self.plot_variants(ax1, performance, avg_type, y_title, start_index)
        self.plot_variants(ax2, performance, avg_type, y_title, start_index, fixed_corr_type="linear")
        self.plot_variants(ax3, performance, avg_type, y_title, start_index, fixed_corr_type="rank")
        self.plot_variants(ax4, performance, avg_type, y_title, start_index, fixed_expl_type="regular")
        self.plot_variants(ax5, performance, avg_type, y_title, start_index, fixed_expl_type="bipoint")

        return (ax1, ax2, ax3, ax4, ax5)