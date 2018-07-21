# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:48:59 2018

@author: alexc
"""

def extract_data(results):
    avg_2_norms = []
    double_std_2_norms = []
    avg_f_norms = []
    double_std_f_norms = []
    max_utilities = []
    max_sampled_utilities = []
    avg_samples_utility = []
    double_std_utility = []
    avg_samples_score = []
    double_std_score = []
    synthetic_datasets = []
    private_data = []
    for result in results:
        avg_2_norms.append(result['avg_2_norm_corr'])
        double_std_2_norms.append(2*result['std_2_norm_corr'])
        avg_f_norms.append(result['avg_f_norm_cov'])
        double_std_f_norms.append(2*result['std_f_norm_cov'])
        max_utilities.append(result['max_utility'])
        max_sampled_utilities.append(result['max_sampled_utility'])
        avg_samples_utility.append(result['sample_utilities_avg'])
        double_std_utility.append(2*result['sample_utilities_std'])
        avg_samples_score.append(result['sample_scores_avg'])
        double_std_score.append(2*result['sample_scores_std'])
        synthetic_datasets.append(result['synthetic_data'])
    test_set = results[0]['test_set']
    private_data = results[0]['private_data']    
    return avg_2_norms, double_std_2_norms, avg_f_norms, double_std_f_norms, max_utilities, max_sampled_utilities, avg_samples_utility, \
            double_std_utility, avg_samples_score, double_std_score, synthetic_datasets, test_set, private_data
