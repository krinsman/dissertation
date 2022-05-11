from scipy.stats import chi2
import numpy as np

def get_observed_counts(strain1, strain2, counts):
    """Return observed counts for a given interaction."""
    strain1_non0_boolean = counts[:,strain1] > 0
    strain2_non0_boolean = counts[:,strain2] > 0
    
    both_strains_non0_boolean = strain1_non0_boolean & strain2_non0_boolean
    not_both_strains_non0_boolean = ~both_strains_non0_boolean
    
    strain1_only_non0_boolean = strain1_non0_boolean & not_both_strains_non0_boolean
    strain2_only_non0_boolean = strain2_non0_boolean & not_both_strains_non0_boolean
    
    number_droplets, number_strains = counts.shape
    
    both_strains_non0 = np.sum(both_strains_non0_boolean)
    strain1_only_non0 = np.sum(strain1_only_non0_boolean)
    strain2_only_non0 = np.sum(strain2_only_non0_boolean)
    both_strains_0 = number_droplets - both_strains_non0 - strain1_only_non0 - strain2_only_non0
    
    return np.asarray([both_strains_0, strain1_only_non0, strain2_only_non0, both_strains_non0])

def get_single_and_double_strain_0_probs(frequencies, rate=2.):
    frequencies = np.asarray(frequencies)
    single_strain_0_probs = np.exp(-(rate*frequencies))
    double_strain_0_probs = np.outer(single_strain_0_probs, single_strain_0_probs)

    return (single_strain_0_probs, double_strain_0_probs)

def get_probabilities(strain1, strain2, single_strain_0_probs, double_strain_0_probs):
    both_strains_0_prob = double_strain_0_probs[strain1, strain2]
    
    strain1_0_prob = single_strain_0_probs[strain1]
    strain2_0_prob = single_strain_0_probs[strain2]
    
    strain1_non0_strain2_0_prob = strain2_0_prob - both_strains_0_prob
    strain2_non0_strain1_0_prob = strain1_0_prob - both_strains_0_prob
    both_strains_non0_prob = (1.-strain1_0_prob)*(1.-strain2_0_prob)
    
    return np.asarray([both_strains_0_prob, strain1_non0_strain2_0_prob, strain2_non0_strain1_0_prob, both_strains_non0_prob])

def get_expected_counts(strain1, strain2, single_strain_0_probs, double_strain_0_probs, number_droplets):
    
    probabilities = get_probabilities(strain1=strain1, strain2=strain2, 
                                        single_strain_0_probs=single_strain_0_probs,
                                        double_strain_0_probs=double_strain_0_probs)
    return number_droplets * probabilities

def get_unconditional_pairwise_pearson_categorical_divergence(strain1, strain2, single_strain_0_probs, double_strain_0_probs, counts, differences=False):
    
    observed_counts = get_observed_counts(strain1=strain1, strain2=strain2, counts=counts)
    
    number_droplets, number_strains = counts.shape
    
    expected_counts = get_expected_counts(strain1=strain1, strain2=strain2, 
                                          single_strain_0_probs=single_strain_0_probs,
                                          double_strain_0_probs=double_strain_0_probs,
                                                      number_droplets=number_droplets)

    divergence = np.sum(((observed_counts - expected_counts)**2)/expected_counts, axis=None)
    return divergence if differences == False else (divergence, observed_counts - expected_counts)
    
def get_all_unconditional_pairwise_pearson_categorical_divergences_and_p_values(counts, frequencies, rate=2., differences=True):
    single_strain_0_probs, double_strain_0_probs = get_single_and_double_strain_0_probs(frequencies=frequencies, rate=rate)
      
    try:
        assert counts.shape[1] == frequencies.size
    except AssertionError:
        counts = counts.T
        assert counts.shape[1] == frequencies.size
    
    divergences = np.zeros((frequencies.size, frequencies.size))
    pvals = np.zeros((frequencies.size, frequencies.size))
    if differences == True:
        difference_vectors = np.zeros((frequencies.size, frequencies.size, 4))
    
    for strain1 in range(frequencies.size):
        for strain2 in range(strain1+1, frequencies.size):
            result = get_unconditional_pairwise_pearson_categorical_divergence(strain1=strain1, strain2=strain2, 
                            single_strain_0_probs=single_strain_0_probs, 
                            double_strain_0_probs=double_strain_0_probs, 
                            counts=counts, differences=differences)
            if differences == False:
                divergence = result
            else:
                divergence, difference_vector = result
            # 4 - 1 = 3
            pval = chi2.sf(divergence, df=3)
            divergences[strain1, strain2] = divergence
            pvals[strain1, strain2] = pval
            
            if differences == True:
                difference_vectors[strain1, strain2, :] = difference_vector
    return (divergences, pvals) if differences == False else (divergences, pvals, difference_vectors)