from scipy.stats import chi2
import numpy as np

def get_conditional_observed_counts(strain1, strain2, counts):
    """Return observed counts for a given interaction."""
    strain1_non0_boolean = counts[:,strain1] > 0
    strain2_non0_boolean = counts[:,strain2] > 0
    
    both_strains_non0_boolean = strain1_non0_boolean & strain2_non0_boolean
    not_both_strains_non0_boolean = ~both_strains_non0_boolean
    
    strain1_only_non0_boolean = strain1_non0_boolean & not_both_strains_non0_boolean
    strain2_only_non0_boolean = strain2_non0_boolean & not_both_strains_non0_boolean
    
    return (np.sum(strain1_only_non0_boolean), np.sum(both_strains_non0_boolean), np.sum(strain2_only_non0_boolean))

def get_single_and_double_strain_non0_probs(frequencies, rate=2.):
    frequencies = np.asarray(frequencies)
    single_strain_0_probs = np.exp(-(rate*frequencies))
    double_strain_0_probs = np.outer(single_strain_0_probs, single_strain_0_probs)

    single_strain_non0_probs = 1. - single_strain_0_probs
    double_strain_non0_probs = 1. - double_strain_0_probs
    return (single_strain_non0_probs, double_strain_non0_probs)

def get_conditional_probabilities(strain1, strain2, single_strain_non0_probs, double_strain_non0_probs):
    strain1_non0_strain2_0_conditional_prob = 1. - (single_strain_non0_probs[strain2]/double_strain_non0_probs[strain1, strain2])
    strain2_non0_strain1_0_conditional_prob = 1. - (single_strain_non0_probs[strain1]/double_strain_non0_probs[strain1, strain2])
    both_non0_conditional_prob = 1. - strain1_non0_strain2_0_conditional_prob - strain2_non0_strain1_0_conditional_prob
    
    return np.asarray([strain1_non0_strain2_0_conditional_prob, both_non0_conditional_prob, 
                       strain2_non0_strain1_0_conditional_prob])

def get_conditional_expected_counts(strain1, strain2, single_strain_non0_probs, double_strain_non0_probs, total_number):
    
    conditional_probabilities = get_conditional_probabilities(strain1=strain1, strain2=strain2, 
                                                              single_strain_non0_probs=single_strain_non0_probs,
                                                              double_strain_non0_probs=double_strain_non0_probs)
    return total_number * conditional_probabilities

def get_conditional_pairwise_pearson_categorical_divergence(strain1, strain2, single_strain_non0_probs, double_strain_non0_probs, counts, differences=False, observed=False):
    
    observed_counts = get_conditional_observed_counts(strain1=strain1, strain2=strain2, counts=counts)
    total_number = np.sum(observed_counts)
    
    if total_number == 0:
        return np.nan if differences == False else (np.nan, np.array([np.nan, np.nan, np.nan]))
    else:
        expected_counts = get_conditional_expected_counts(strain1=strain1, strain2=strain2, 
                                                          single_strain_non0_probs=single_strain_non0_probs,
                                                          double_strain_non0_probs=double_strain_non0_probs,
                                                          total_number=total_number)

        divergence = np.sum(((observed_counts - expected_counts)**2)/expected_counts, axis=None)
        
        results_dict = {'divergence': divergence}
        if differences:
            results_dict['difference_vector'] = observed_counts - expected_counts
        if observed:
            results_dict['observed_counts'] = observed_counts
        return results_dict
    
def get_all_conditional_pairwise_pearson_categorical_divergences_and_p_values(counts, frequencies, rate=2., differences=False, observed=False):
    single_strain_non0_probs, double_strain_non0_probs = get_single_and_double_strain_non0_probs(frequencies=frequencies, rate=rate)
      
    try:
        assert counts.shape[1] == frequencies.size
    except AssertionError:
        counts = counts.T
        assert counts.shape[1] == frequencies.size
    
    divergences = np.zeros((frequencies.size, frequencies.size))
    pvals = np.zeros((frequencies.size, frequencies.size))
    if differences == True:
        difference_vectors = np.zeros((frequencies.size, frequencies.size, 3))
    if observed == True:
        observed_counts = np.zeros((frequencies.size, frequencies.size, 3))
    
    for strain1 in range(frequencies.size):
        for strain2 in range(strain1+1, frequencies.size):
            results_dict = get_conditional_pairwise_pearson_categorical_divergence(strain1=strain1, strain2=strain2, 
                            single_strain_non0_probs=single_strain_non0_probs, 
                            double_strain_non0_probs=double_strain_non0_probs, counts=counts, differences=differences, observed=observed)
            divergence = results_dict['divergence']

            pval = chi2.sf(divergence, df=2)
            divergences[strain1, strain2] = divergence
            pvals[strain1, strain2] = pval
            
            if differences:
                difference_vectors[strain1, strain2, :] = results_dict['difference_vector']
            if observed:
                observed_counts[strain1, strain2, :] = results_dict['observed_counts']
    results_dict = {'divergences': divergences, 'pvals': pvals}
    if differences:
        results_dict['difference_vectors'] = difference_vectors
    if observed:
        results_dict['observed_counts'] = observed_counts
    return results_dict