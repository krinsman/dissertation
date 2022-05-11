import scipy
import numpy as np
import seaborn as sns

def initialize_reflected_kde(kde_bounds=(0,1)):
    lb, ub = kde_bounds
    # lb = lower bound, ub = upper bound

    ### Handle KDE plots
    kde_evaluate = scipy.stats.gaussian_kde.evaluate

    def reflected_kde_evaluate(self, x):    

        if (len(x) == self.d) and (type(x) == list) and (self.d == 2):
            # inside horizontal and vertical boundaries
            val = np.where(((x[0]>=lb)&(x[1]>=lb))&((x[0]<=ub)&(x[1]<=ub)), kde_evaluate(self, x), 0)
            # to left of horizontal boundaries, inside vertical boundaries
            val += np.where(((x[0]>=lb)&(x[1]>=lb))&((x[0]<=ub)&(x[1]<=ub)), kde_evaluate(self, [lb-x[0],x[1]]), 0)
            # inside horizontal boundaries, below vertical boundaries
            val += np.where(((x[0]>=lb)&(x[1]>=lb))&((x[0]<=ub)&(x[1]<=ub)), kde_evaluate(self, [x[0],lb-x[1]]), 0)
            # to left of horizontal boundaries, below vertical boundaries
            val += np.where(((x[0]>=lb)&(x[1]>=lb))&((x[0]<=ub)&(x[1]<=ub)), kde_evaluate(self, [lb-x[0],lb-x[1]]), 0)
            # inside horizontal boundaries, above vertical boundaries
            val += np.where(((x[0]>=lb)&(x[1]>lb))&((x[0]<=ub)&(x[1]<=ub)), kde_evaluate(self, [x[0],ub-(x[1]-ub)]), 0)
            # to right of horizontal boundaries, inside vertical boundaries
            val += np.where(((x[0]>lb)&(x[1]>=lb))&((x[0]<=ub)&(x[1]<=ub)), kde_evaluate(self, [ub-(x[0]-ub),x[1]]), 0)
            # to right of horizontal boundaries, above vertical boundaries
            val += np.where(((x[0]>lb)&(x[1]>lb))&((x[0]<=ub)&(x[1]<=ub)), kde_evaluate(self, [ub-(x[0]-ub),ub-(x[1]-ub)]), 0)
            # to left of horizontal boundaries, above vertical boundaries
            val += np.where(((x[0]>=lb)&(x[1]>lb))&((x[0]<=ub)&(x[1]<=ub)), kde_evaluate(self, [lb-x[0],ub-(x[1]-ub)]), 0)
            # to right of horizontal boundaries, below vertical boundaries
            val += np.where(((x[0]>lb)&(x[1]>=lb))&((x[0]<=ub)&(x[1]<=ub)), kde_evaluate(self, [ub-(x[0]-ub),lb-x[1]]), 0)
        elif self.d == 1:
            val = np.where((x>=lb)&(x<=ub), kde_evaluate(self, x), 0)
            val += np.where((x>=lb)&(x<=ub), kde_evaluate(self, lb-x), 0)
            val += np.where((x>lb)&(x<=ub), kde_evaluate(self, ub-(x-ub)), 0)
        return val

    scipy.stats.kde.gaussian_kde.__call__ = reflected_kde_evaluate
    
    ### Handle Violin Plots
    
    # https://github.com/mwaskom/seaborn/issues/525#issuecomment-97651992
    fit_kde_func = sns.categorical._ViolinPlotter.fit_kde

    def reflected_once_kde(self, x, bw):
        kde, bw_used = fit_kde_func(self, x, bw)

        kde_evaluate = kde.evaluate

        def truncated_kde_evaluate(x):
            val = np.where((x >= lb) & (x <= ub), kde_evaluate(x), 0)
            val += np.where((x >= lb) & (x <= ub), kde_evaluate(lb - x), 0)
            val += np.where((x > lb) & (x <= ub), kde_evaluate(ub - (x - ub)), 0)
            return val

        kde.evaluate = truncated_kde_evaluate
        return kde, bw_used

    sns.categorical._ViolinPlotter.fit_kde = reflected_once_kde