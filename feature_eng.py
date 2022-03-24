import pandas as pd
import numpy as np
from statsmodels.stats.power import tt_ind_solve_power
from collections import Counter

def get_intial_analysis(data, 
                        target_var = 'increase_in_value',           
                        potential_feature = ''):      
    agg_dictionary = {target_var:['sum', 'count', 'mean']}          
    base_rate = np.mean(data[target_var])
    agg_df = data.groupby(potential_feature).agg(agg_dictionary)[target_var]   
    agg_df['Abs. diff. from mean'] = (agg_df['mean'] - base_rate).apply(abs)
    agg_df = agg_df.sort_values('Abs. diff. from mean')
    agg_df.columns = [col.capitalize() for col in list(agg_df)]
    return agg_df

def get_power_df(data, feature = '', target_var = ''): 
    power_df = data.groupby(feature)[target_var].agg(Counter)
    return power_df

def get_power_stats(data, feature = '', target_var = 'increase_in_value'):  
    power_df = get_power_df(data, feature = feature, target_var = target_var)
    directional_power_stats = {}
    for group, B_group in power_df.iteritems():
        # A group is everything but what we're looking into 
        A_group = power_df.drop(group)
        A_neg_sum = sum([_counter[0] for _counter in A_group.values])
        A_pos_sum = sum([_counter[1] for _counter in A_group.values])

        B_neg_sum = B_group[0]; B_pos_sum = B_group[1]
        n_A = sum([A_neg_sum, A_pos_sum])
        n_B = sum([B_neg_sum, B_pos_sum])

        mu_B = B_pos_sum/n_B

        if n_A > 0:
            # To avoid divide by zero errors                
            mu_A = A_pos_sum/n_A
            A_pos_sq_diff = A_pos_sum*((1 - mu_A)**2)
            A_neg_sq_diff = A_neg_sum*((0 - mu_A)**2)

            B_pos_sq_diff = B_pos_sum*((1 - mu_B)**2)
            B_neg_sq_diff = B_neg_sum*((0 - mu_B)**2)

            var_A = (A_pos_sq_diff + A_neg_sq_diff)/n_A
            var_B = (B_pos_sq_diff + B_neg_sq_diff)/n_B

            std_A = var_A**(1/2)
            std_B = var_B**(1/2)

            std_A = 1e-300 if std_A == 0 else std_A
            std_B = 1e-300 if std_B == 0 else std_B

            effect_size = (mu_B - mu_A) / ((n_A * std_A + n_B * std_B) / (n_A + n_B))

            if n_B:
                power_stat = tt_ind_solve_power(
                                            effect_size = effect_size, 
                                            nobs1 = n_A, 
                                            alpha = 0.10, 
                                            power = None, 
                                            ratio = n_B/n_A, 
                                            alternative = 'two-sided')

                # Needs to be negative if mean is in the opposite direction
                power_stat = -power_stat if mu_B < mu_A else power_stat
                directional_power_stats[group] = power_stat
    
    directional_power_stats = pd.DataFrame(
                                        directional_power_stats.items(),
                                        columns = [feature, f'dir_power_stat_{feature}'])

    return directional_power_stats.sort_values(f'dir_power_stat_{feature}').reset_index(drop = True)
