import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# WARNING: horribly monkey patch follows below, please forgive me if you can
# see explanation: https://stackoverflow.com/a/52818574
def initialize_kde(kde_bounds):
    """The following leads to slightly more accurate/less misleading VIOLIN plots when mass is concentrated near boundaries, as happens often in this notebook. (Doesn't also affect the KDE used in sns.kdeplot because that would be convenient.) I don't really like the programming with side effects though but whatever I guess.
    
    Note that in general this needs to be paired with `cut=0` argument/parameter
    in sns.violinplot in order to not look bad
    """
    # https://github.com/mwaskom/seaborn/issues/525#issuecomment-97651992
    fit_kde_func = sns.categorical._ViolinPlotter.fit_kde

    lb, ub = kde_bounds

    def reflected_once_kde(self, x, bw):
        kde, bw_used = fit_kde_func(self, x, bw)

        kde_evaluate = kde.evaluate

        def truncated_kde_evaluate(x):
            """basically idea seems to be to allocate sum of probability that original Gaussian kernel gave to $-x$ and $+x$ all to $+x$, so that we don't lose probability when cutting off everything beyond 0."""
            val = np.where((x >= lb) & (x <= ub), kde_evaluate(x), 0)
            val += np.where((x >= lb) & (x <= ub), kde_evaluate(lb - x), 0)
            val += np.where((x > lb) & (x <= ub), kde_evaluate(ub - (x - ub)), 0)
            return val

        kde.evaluate = truncated_kde_evaluate
        return kde, bw_used

    sns.categorical._ViolinPlotter.fit_kde = reflected_once_kde

def log_score_formatter(y, pos):
    if y == 1:
        return '90%'
    elif y == 2:
        return '99%'
    elif y == 3:
        return '99.9%'
    elif y == -np.log(0.5)/np.log(10):
        return '50%'

def add_experiments(mode_indices, mode_definitions):
        mode_indices = {choice:index+2 for choice, index in mode_indices.items()}
        mode_indices["experiment_type"] = 0
        mode_indices["control_type"] = 1    

        setup_variants = {0: "gluttonous", 1: "picky"}
        mode_definitions = [setup_variants, setup_variants, *mode_definitions]
        return mode_indices, mode_definitions
    
reverse_dict = lambda dictionary: {value:key for key, value in dictionary.items()}
    
class Plotter:
    percentages = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0", "2.0", "5.0", "10.0"]
    log_score_formatter = FuncFormatter(log_score_formatter)
    important_log_scores = [-np.log(0.5)/np.log(10),1,2,3]
    
    number_simulations = 100
    
    mode_indices = None
    mode_definitions = None
    
    def __init__(self, base_result_filename, results_dir, kde_bounds=None):
        self.results_dir = results_dir
        self.base_result_filename = base_result_filename
        
        self.methods_shape = tuple([len(definitions) for definitions in self.mode_definitions])
                
        if kde_bounds is not None:
            initialize_kde(kde_bounds)
          
    @classmethod    
    def get_choices(cls, mode_name):
        mode_index = cls.mode_indices[mode_name]
        return cls.mode_definitions[mode_index].values()

    def load_performance_results(self, metric_name):
        raise NotImplementedError
        
    @classmethod
    def variant_to_multiindex(cls, **kwargs):
        assert any(variant_type is not None for variant_type in kwargs.items())

        relevant_mode_indices = [cls.mode_indices[variant] for variant in kwargs.keys()]

        multiindex = tuple([cls.mode_definitions_r[index][variant_type]
                           for index, variant_type in
                           zip(relevant_mode_indices, kwargs.values())])
        return multiindex

    def plot_metric(self,
        ax, array_slice, color="green", title="", label="", y_title="", start_index=0
    ):
        data = [array_slice[:, i] for i in range(len(self.percentages))]
        data = data[start_index:]
        # get > -np.inf and < np.inf in one go (I think), plus also not NaN
        data = [vector[np.isfinite(vector)] for vector in data]
        sns.violinplot(ax=ax, data=data, cut=0, color=color, linewidth=2)
        # https://stackoverflow.com/a/62598287, https://stackoverflow.com/a/3453101
        for violin in ax.collections[-3 * len(self.percentages) :][::2]:
            violin.set_alpha(0.5)
        medians = [np.median(vector) for vector in data]
        sns.lineplot(ax=ax, data=medians, color=color, label=label)
        upper_quartiles = [np.percentile(vector, 75) for vector in data]
        lower_quartiles = [np.percentile(vector, 25) for vector in data]
        x = np.arange(start_index, len(self.percentages)) - start_index
        ax.fill_between(x, upper_quartiles, lower_quartiles, color=color, alpha=0.3)
        ax.set_title(title)
        ax.set_ylabel(y_title)
        ax.set_xticklabels(
            ["{}%".format(percentage) for percentage in self.percentages][start_index:]
        )
        return ax

    def add_labels(self, ax, title="", y_title="", start_index=0):
        ax.set_ylabel(y_title)
        if start_index > 0:
            ax.set_title('{} - Highest {} Relative Abundances'.format(title, len(self.percentages) - start_index))
        else:
            ax.set_title(title)
        ax.legend()

    def annotate_important_log_scores(self, ax, y_title=''):
        axb = ax.twinx()
        axb.set_ylim(ax.get_ylim())

        axb.set_yticks(self.important_log_scores)

        axb.yaxis.set_major_formatter(self.log_score_formatter)
        for y in self.important_log_scores:
            ax.axhline(y=y, color='gray', linestyle='--')
        ax.legend(loc='lower right')
        axb.set_ylabel(y_title)