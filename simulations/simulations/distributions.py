import numpy as np

def dirichlet_multinomial(concentration, frequencies, number_cells, rng):
    frequencies = np.asarray(frequencies)
    S = frequencies.size
    local_frequencies = rng.dirichlet(concentration*S*frequencies)
    cell_distribution_vector = rng.multinomial(n=number_cells, pvals=local_frequencies)
    return cell_distribution_vector

# Compound Poisson Dirichlet Multinomial
def CPDM(concentration, frequencies, rate, rng):
    number_cells = rng.poisson(lam=rate)
    cell_distribution_vector = dirichlet_multinomial(concentration=concentration, frequencies=frequencies,
                                                    number_cells=number_cells, rng=rng)
    return cell_distribution_vector

def custom_negative_binomial(concentration, number_species, rate, rng):
    unnorm_concentration = concentration * number_species
    paramvals = unnorm_concentration / (unnorm_concentration + rate)
    number_cells = rng.negative_binomial(n=unnorm_concentration, p=paramvals)
    return number_cells

# compound negative binomial dirichlet multinomial
def CNBDM(concentration, frequencies, rate, rng):
    frequencies = np.asarray(frequencies)
    number_cells = custom_negative_binomial(concentration=concentration, 
                                            number_species=frequencies.size, 
                                            rate=rate, rng=rng)
    cell_distribution_vector = dirichlet_multinomial(concentration=concentration, frequencies=frequencies,
                                                    number_cells=number_cells, rng=rng)
    return cell_distribution_vector

# this is CPM - compound Poisson Multinomial
def CPM(poisson_rate, multinomial_frequencies, rng):
    number_cells = rng.poisson(poisson_rate)
    initial_droplet = rng.multinomial(number_cells, multinomial_frequencies)
    return initial_droplet

### Interaction distributions

def interaction_dirichlet(concentration, frequencies, interaction_matrix, rng, 
                          pre_link=lambda x: x, post_link=np.exp):
    """`post_link` needs to map real numbers to non-negative real numbers.
    `pre_link` and `post_link` together need to satisfy that 
    `(np.dot(a,x) <= np.dot(b,x)) == True` implies that
    `post_link(np.dot(a,pre_link(x))) <= post_link(np.dot(b,pre_link(x))) == True`.
    When `pre_link` is the identity, as is the default, it suffices for `post_link` to
    be any non-decreasing function, e.g. `np.exp` or `lambda x: np.maximum(x,0)`."""
    frequencies = np.asarray(frequencies)
    shape_params = (concentration * frequencies.size)*frequencies
    local_counts = rng.standard_gamma(shape=shape_params)
    # transpose b/c our convention is that entry (i,j) is effect of i on j
    transformed_local_counts = post_link(interaction_matrix.T @ (pre_link(local_counts)))
    local_frequencies = transformed_local_counts / np.sum(transformed_local_counts)
    return local_frequencies

def interaction_dirichlet_multinomial(concentration, frequencies, interaction_matrix,
                                     number_cells, rng, pre_link=lambda x: x, post_link=np.exp):
    """See docstring for `interaction_dirichlet` regarding `pre_link` and `post_link`."""
    local_frequencies = interaction_dirichlet(concentration=concentration, frequencies=frequencies,
                                             interaction_matrix=interaction_matrix, rng=rng,
                                             pre_link=pre_link, post_link=post_link)
    cell_distribution_vector = rng.multinomial(n=number_cells, pvals=local_frequencies)
    return cell_distribution_vector

def CPIDM(concentration, frequencies, interaction_matrix, rate, rng, 
          pre_link=lambda x: x, post_link=np.exp):
    """See docstring for `interaction_dirichlet` regarding `pre_link` and `post_link`"""
    number_cells = rng.poisson(lam=rate)
    cell_distribution_vector = interaction_dirichlet_multinomial(concentration=concentration,
                                                                frequencies=frequencies,
                                                                interaction_matrix=interaction_matrix,
                                                                number_cells=number_cells, rng=rng,
                                                                pre_link=pre_link, post_link=post_link)
    return cell_distribution_vector