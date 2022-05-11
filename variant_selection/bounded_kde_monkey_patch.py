import scipy
import numpy as np

kde_evaluate = scipy.stats.gaussian_kde.evaluate

def truncated_kde_evaluate(self, x):    
    lb = 0
    ub = 1
        
    # Only need this part for two dimensional distributions
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
    # This suffices as monkey patch for one dimensional distributions
    elif self.d == 1:
        val = np.where((x>=lb)&(x<=ub), kde_evaluate(self, x), 0)
        val += np.where((x>=lb)&(x<=ub), kde_evaluate(self, lb-x), 0)
        val += np.where((x>lb)&(x<=ub), kde_evaluate(self, ub-(x-ub)), 0)
    return val

scipy.stats.kde.gaussian_kde.__call__ = truncated_kde_evaluate