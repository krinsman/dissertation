import numpy as np

def get_agnostic_weights(relevant_counts, strain_agnostic_M):
    agnostic_weights = np.zeros(strain_agnostic_M).astype(int)
    for i in range(strain_agnostic_M):
        agnostic_weights[i] = np.sum(relevant_counts > i)
    # idea is to use commutativity of addition and group redundant terms
    # cf. `v` vector from https://arxiv.org/abs/1405.0099
    return agnostic_weights

def get_specific_weights(relevant_droplets, strain_specific_M):
    number_droplets, number_strains = relevant_droplets.shape
    
#     screened_batches = {}
#     # don't like this code because there should be some optimization
#     # based on the fact that e.g. x > 5 implies x > 4 -- calculating
#     # from scratch for every i seems very wasteful but not fluent 
#     # enough in NumPy to know/think of a better way
#     for i in range(strain_specific_M):
#         screened_batches[i] = relevant_droplets > i
#     # anyway idea is to use commutativity of addition and group redundant terms
#     # cf. `U` matrix from https://arxiv.org/abs/1405.0099

    specific_weights = np.zeros((strain_specific_M, number_strains)).astype(int)

#     for i in range(strain_specific_M):
#         specific_weights[i,:] = np.sum(screened_batches[i], axis=0)
    
    for i in range(strain_specific_M):
        screened_batches = relevant_droplets > i
        specific_weights[i,:] = np.sum(screened_batches, axis=0)
        del(screened_batches)
    
    return specific_weights

def get_DM_score_function(batch, freqs):
    
    number_droplets, number_strains = batch.shape
    
    counts = np.sum(batch, axis=1)
    relevant_indices = counts >= 2
    relevant_droplets = batch[relevant_indices, :]
    relevant_counts = counts[relevant_indices]
    
    strain_agnostic_M = np.max(relevant_counts)
    agnostic_weights = get_agnostic_weights(relevant_counts, strain_agnostic_M)
    
    strain_specific_M = np.max(relevant_droplets)
    specific_weights = get_specific_weights(relevant_droplets, strain_specific_M)
    
    def agnostic_score_function(conc):
        values = float(number_strains) / ( (conc * number_strains) + np.arange(strain_agnostic_M))
        values = -agnostic_weights*values
        return np.sum(values)

    def specific_score_function(conc):
        increments = np.arange(strain_specific_M).reshape((-1,1))
        # row vector with length same as row length of specific_weights, number strains
        strain_concs = conc * number_strains * freqs.reshape((1,-1))
        # using broadcasting to get a matrix the same size as specific_weights
        # values = float(number_strains) / (strain_concs + increments)
        values = (number_strains * freqs.reshape((1,-1))) / (strain_concs + increments)
        # old code did not weight each strain's score term by its frequency but should have
        # makes big difference -- otherwise score is wrong and increases without bound
        weighted_values = specific_weights * values
        return np.sum(weighted_values)

    def score_function(conc):
        return agnostic_score_function(conc) + specific_score_function(conc)
    
#     def agnostic_jacobian(conc):
#         values = float(number_strains**2) / ( (conc * number_strains) + np.arange(strain_agnostic_M))**2
#         values = agnostic_weights*values
#         return np.sum(values)
    
#     def specific_jacobian(conc):
#         increments = np.arange(strain_specific_M).reshape((-1,1))
#         # row vector with length same as row length of specific_weights, number strains
#         strain_concs = conc * number_strains * freqs.reshape((1,-1))
#         # using broadcasting to get a matrix the same size as specific_weights
#         #values = float(number_strains**2) / (strain_concs + increments)**2
#         values = (number_strains**2 * (freqs**2).reshape((1,-1))) / (strain_concs + increments)
#         # old code did not weight each strain's term by its frequency squared but should have
#         weighted_values = -specific_weights * values
#         return np.sum(weighted_values)
    
#     def jacobian(conc):
#         """
#         `fsolve` complains if Jacobian returns a scalar and not a NumPy array
#         never mind that there's no real difference in the 1D case like this
#         `Numpy : ValueError: object of too small depth for desired array`
#         so I agree that turning this into a one element NumPy array is dumb but whatever
#         scipy docs say about the `fprime` argument:
#         `A function to compute the Jacobian of func with derivatives across the rows.`
#         so I guess the problem is that scalars don't have "rows" 
#         (although neither does really a 1D NumPy array? also I thought I had used
#         fsolve before with scalar functions like this with the Jacboain returning a scalar)
#         """
#         return np.array([agnostic_jacobian(conc) + specific_jacobian(conc)])
    
    return score_function#, jacobian


def get_NB_score_function(counts, mean_count, number_droplets, number_strains):
    max_count = np.max(counts)
    
    weights = np.zeros(max_count)
    
    for m in range(max_count):
        relevant_counts = counts > m
        weights[m] = np.sum(relevant_counts)
        # should be able to do this b/c x > m + 1 implies x > m
        counts = counts[relevant_counts]
        # ^for computational efficiency when max_count large and counts long vector
        
    def score_function(concentration):
        increments = np.arange(max_count)
        denominators = (concentration * number_strains) + increments
        weighted_values = weights / denominators
        
        result = -number_droplets * np.log(1 + (mean_count/(concentration * number_strains)))
        result += np.sum(weighted_values)
        return result
    
    return score_function