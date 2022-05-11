import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from string import capwords
from itertools import product

from .plot_utils import Plotter

def log_value_formatter(y, pos):
    if y == -1:
        return '10%'
    elif y == -2:
        return '1%'
    elif y == -3:
        return '0.1%'
    elif y == -6:
        return 'lightning strike'
    elif y == np.log(0.5)/np.log(10):
        return '50%'

class phenopathVariantsPlotter(Plotter):
    
    important_log_scores = [-1,-2,-3,-6,np.log(0.5)/np.log(10)]
    log_score_formatter = FuncFormatter(log_value_formatter)
    
    mode_indices = {"covar_choice": 0, "scale_choice": 1, "censor_choice": 2}
    mode_definitions = [{0:"binary_covariates",1:"count_covariates", 2:"log_count_covariates"}, 
                    {0:"scaled",1:"unscaled"},
                    {0:"uncensored",1:"censored"}]
    
    def load_performance_results(self, metric_name):
        base_filename = "{}/pct{{}}/{}".format(self.results_dir, self.base_result_filename)
        self.filenames = [base_filename.format(percentage) for percentage in self.percentages]
        
        performance = np.zeros((*self.methods_shape, self.number_simulations, len(self.percentages)))

        for index, filename in enumerate(self.filenames):
            npzfile = np.load(filename)
            performance[..., index] = npzfile[metric_name]

        return performance
    
    def censor_choices_side_by_side(self, performance, covar_choice, scale_choice, y_title="", start_index=0):
        fig, ax = plt.subplots(figsize=(20, 10))

        multiindex = self.variant_to_multiindex(covar_choice=covar_choice, scale_choice=scale_choice)
        performance = performance[(*multiindex, ...)]
        title = '{} {}'.format(scale_choice.capitalize(), capwords(covar_choice.replace('_', ' ')))

        colors = {"censored": "red", "uncensored": "green"}
        for censor_choice in self.get_choices("censor_choice"):
            censor_choice_code = self.variant_to_multiindex(censor_choice=censor_choice)[0]
            color = colors[censor_choice]
            self.plot_metric(ax, performance[censor_choice_code,...], color=color, 
                             start_index=start_index, label=censor_choice.capitalize())
        self.add_labels(ax, title=title, y_title=y_title, start_index=start_index)
        return ax

    def get_color_name(self, covar_choice, scale_choice):
        if covar_choice == "binary_covariates":
            if scale_choice == "unscaled":
                return "deepskyblue"
            elif scale_choice == "scaled":
                return "dodgerblue"
        elif covar_choice == "count_covariates":
            if scale_choice == "unscaled":
                return "yellow"
            elif scale_choice == "scaled":
                return "gold"
        elif covar_choice == "log_count_covariates":
            if scale_choice == "unscaled":
                return "greenyellow"
            elif scale_choice == "scaled":
                return "yellowgreen"
    
    def plot_variant(self, covar_choice, scale_choice, ax, performance, censor_choice, start_index=0):        
        variant_multiindex = self.variant_to_multiindex(covar_choice=covar_choice, scale_choice=scale_choice)
        performance = performance[(*variant_multiindex, ...)]
        label = '{} {}'.format(capwords(covar_choice.replace('_', ' ')), scale_choice.capitalize())
        color = self.get_color_name(covar_choice, scale_choice)
        censor_choice_code = self.variant_to_multiindex(censor_choice=censor_choice)[0]
        self.plot_metric(ax, performance[censor_choice_code, ...], color=color, label=label, start_index=start_index)
        return ax
    
    def plot_variants(self, ax, performance, censor_choice, y_title, start_index, fixed_covar_choice="", fixed_scale_choice=""):
        assert (not fixed_covar_choice) or (not fixed_scale_choice)
        for covar_choice, scale_choice in product(self.get_choices("covar_choice"), self.get_choices("scale_choice")):
            is_covar_choice_match = (not fixed_covar_choice) or (covar_choice == fixed_covar_choice)
            is_scale_choice_match = (not fixed_scale_choice) or (scale_choice == fixed_scale_choice)
            if is_covar_choice_match and is_scale_choice_match:
                self.plot_variant(covar_choice, scale_choice, ax, performance, censor_choice, start_index)
            
        title = '{}'.format(censor_choice.capitalize())
        if fixed_covar_choice:
            title = '{} - {} Only'.format(title, fixed_covar_choice.capitalize())
        if fixed_scale_choice:
            title = '{} - {} Only'.format(title, fixed_scale_choice.capitalize())
        self.add_labels(ax, title=title, y_title=y_title, start_index=start_index)
        

    # https://stackoverflow.com/a/37232760
    def covar_and_scale_choices_side_by_side(self,
        performance, censor_choice="uncensored", y_title="", start_index=0
    ):
        nrows = 1 + len(self.get_choices("covar_choice")) + len(self.get_choices("scale_choice"))
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=nrows, figsize=(20, 10*nrows))
        
        self.plot_variants(ax1, performance, censor_choice, y_title, start_index)
        self.plot_variants(ax2, performance, censor_choice, y_title, start_index, fixed_covar_choice="binary_covariates")
        self.plot_variants(ax3, performance, censor_choice, y_title, start_index, fixed_covar_choice="count_covariates")
        self.plot_variants(ax4, performance, censor_choice, y_title, start_index, fixed_covar_choice="log_count_covariates")
        self.plot_variants(ax5, performance, censor_choice, y_title, start_index, fixed_scale_choice="unscaled")
        self.plot_variants(ax6, performance, censor_choice, y_title, start_index, fixed_scale_choice="scaled")

        return (ax1, ax2, ax3, ax4, ax5, ax6)