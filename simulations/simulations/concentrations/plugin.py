import numpy as np

def get_plugin_mult_cov(n, emp_freqs):
    plugin_mult_cov = -n*np.outer(emp_freqs, emp_freqs)
    np.fill_diagonal(plugin_mult_cov, n*emp_freqs*(1.-emp_freqs))
    return plugin_mult_cov

def get_emp_cov(droplets_stratum):
    return np.cov(droplets_stratum.T, bias=True)

def get_emp_overdisp(emp_cov, plugin_mult_cov):
    diff_matrix = emp_cov - plugin_mult_cov
    emp_overdisp = diff_matrix / (plugin_mult_cov + (plugin_mult_cov == 0))
    return emp_overdisp

def get_plugin_categorical_concentration(batch):
    
    number_droplets, number_strains = batch.shape
    
    counts = np.sum(batch, axis=1)
    cell_num_levels = np.unique(counts)
    stratified_droplets = {cell_num_level: batch[counts == cell_num_level, :] 
                           for cell_num_level in cell_num_levels}
    nonzero_droplets_count = np.sum([stratum.shape[0] 
                                     for cell_num, stratum in stratified_droplets.items()
                                    if cell_num >= 1])
    emp_freq_weights = {cell_num: stratum.shape[0]/nonzero_droplets_count 
                        for cell_num, stratum in stratified_droplets.items()
                       if cell_num >= 1}
    emp_means_strata = {cell_num: np.mean(stratum, axis=0) 
                        for cell_num, stratum in stratified_droplets.items()
                       if cell_num >= 1}
    emp_freqs_strata = {cell_num: emp_means_stratum/cell_num 
                        for cell_num, emp_means_stratum 
                        in emp_means_strata.items()}
    emp_freqs = np.sum([emp_freq_weight*emp_freqs_stratum 
            for emp_freq_weight, emp_freqs_stratum 
            in zip(emp_freq_weights.values(), emp_freqs_strata.values())], axis=0)
    multi_cell_droplets_count = np.sum([stratum.shape[0] 
                                        for cell_num, stratum 
                                        in stratified_droplets.items()
                                       if cell_num >= 2])
    emp_overdisp_weights = {cell_num: stratum.shape[0]/multi_cell_droplets_count 
                            for cell_num, stratum in stratified_droplets.items()
                            if cell_num >= 2}
    plugin_mult_covs = {cell_num: get_plugin_mult_cov(cell_num, emp_freqs) for cell_num 
                        in cell_num_levels if cell_num >= 2}
    emp_covs = {cell_num: get_emp_cov(droplets_stratum)
           for cell_num, droplets_stratum in stratified_droplets.items()
                if cell_num >= 2}
    emp_overdisp_mats = {cell_num: get_emp_overdisp(emp_covs[cell_num], plugin_mult_covs[cell_num])
                        for cell_num in cell_num_levels if cell_num >= 2}
    adjusted_emp_overdisp_mats = {cell_num: emp_overdisp_mat/(cell_num-1) 
                                  for cell_num, emp_overdisp_mat 
                                  in emp_overdisp_mats.items()}
    adjusted_emp_overdisp_mat = np.sum([emp_overdisp_weights[cell_num]*adjusted_emp_overdisp_mats[cell_num]
                                       for cell_num in cell_num_levels if cell_num >= 2], axis=0)
    # mean is L2 projection onto space of matrices where all entries have same value
    # but use weighted mean since values estimated from more frequent strains/counts 
    # are more reliable. honestly don't know how to mathematically justify
    emp_freq_weighted_AEOM = adjusted_emp_overdisp_mat * np.outer(emp_freqs, emp_freqs)
    
    temp_val = np.sum(emp_freq_weighted_AEOM)
    
    if max(temp_val, 0) == 0:
        return np.inf
    # values of `temp_val` greater than 1 did not occur in practice for the simulations
    # so the following two lines were not used and thus have not been tested
    elif min(temp_val, 1) == 1:
        return 0.
    
    plugin_zeta = ((1. / temp_val ) - 1)/number_strains
    return plugin_zeta

def get_plugin_density_concentration(batch):
    number_droplets, number_strains = batch.shape
    counts = np.sum(batch, axis=1)
    
    estimated_rate = np.mean(counts)
    
    overdispersion = max(((np.var(counts) - estimated_rate)/ estimated_rate), 0)
    if overdispersion == 0:
        return np.inf
    
    adjusted_overdispersion = (number_strains * overdispersion) / estimated_rate
    return 1. / adjusted_overdispersion