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
    sample_utilities = []
    max_sampled_utilities = []
    min_sampled_utilities = []
    avg_samples_utility = []
    double_std_utility = []
    avg_samples_score = []
    double_std_score = []
    synthetic_datasets = []
    private_data = []
    delta_opt_avg = []
    delta_opt_best = []
    delta_opt_worst = []
    for result in results:
        avg_2_norms.append(result['avg_2_norm_corr'])
        double_std_2_norms.append(2*result['std_2_norm_corr'])
        avg_f_norms.append(result['avg_f_norm_cov'])
        double_std_f_norms.append(2*result['std_f_norm_cov'])
        max_utilities.append(result['max_utility'])
        sample_utilities.append(result['sample_utilities'])
        max_sampled_utilities.append(result['max_sampled_utility'])
        min_sampled_utilities.append(result['min_sampled_utility'])
        avg_samples_utility.append(result['sample_utilities_avg'])
        double_std_utility.append(2*result['sample_utilities_std'])
        avg_samples_score.append(result['sample_scores_avg'])
        double_std_score.append(2*result['sample_scores_std'])
        synthetic_datasets.append(result['synthetic_data'])
        delta_opt_avg.append(result['delta_opt_avg'])
        delta_opt_best.append(result['delta_opt_best'])
        delta_opt_worst.append(result['delta_opt_worst'])
    test_set = results[0]['test_set']
    private_data = results[0]['private_data']    
    return avg_2_norms, double_std_2_norms, avg_f_norms, double_std_f_norms, max_utilities, sample_utilities, max_sampled_utilities, min_sampled_utilities, \
            avg_samples_utility, double_std_utility, avg_samples_score, double_std_score, synthetic_datasets, delta_opt_avg,\
            delta_opt_best, delta_opt_worst, test_set, private_data

def initialise_netmech_containers(epsilon_vec):
    avg_2_norms = {key: [] for key in epsilon_vec}
    double_std_2_norms = {key: [] for key in epsilon_vec}
    avg_f_norms = {key: [] for key in epsilon_vec}
    double_std_f_norms = {key: [] for key in epsilon_vec}
    max_utilities = {key: [] for key in epsilon_vec}
    sample_utilities = {key: [] for key in epsilon_vec}
    max_sampled_utilities = {key: [] for key in epsilon_vec}
    min_sampled_utilities = {key: [] for key in epsilon_vec}
    avg_samples_utility = {key: [] for key in epsilon_vec}
    double_std_utility = {key: [] for key in epsilon_vec}
    avg_samples_score = {key: [] for key in epsilon_vec}
    double_std_score = {key: [] for key in epsilon_vec}
    synthetic_datasets_vec = {key: [] for key in epsilon_vec}
    delta_opt_avg = {key: [] for key in epsilon_vec}
    delta_opt_best = {key: [] for key in epsilon_vec}
    delta_opt_worst = {key: [] for key in epsilon_vec}
    test_set = {key: [] for key in epsilon_vec}
    private_data = {key: [] for key in epsilon_vec}
    return avg_2_norms, double_std_2_norms, avg_f_norms, double_std_f_norms, max_utilities, sample_utilities, max_sampled_utilities, min_sampled_utilities, \
            avg_samples_utility, double_std_utility, avg_samples_score, double_std_score, synthetic_datasets_vec, delta_opt_avg,\
            delta_opt_best, delta_opt_worst, test_set, private_data

def initialise_netmech_reg_containers(epsilon_vec):
    net_mech_reg_coefs = {key: [] for key in epsilon_vec}
    predictive_errs_netmech = {key: [] for key in epsilon_vec}
    min_predictive_errs_netmech = {key: [] for key in epsilon_vec}
    mean_predictive_errs_netmech = {key: [] for key in epsilon_vec}
    double_std_predictive_errs_netmech = {key: [] for key in epsilon_vec}
    return net_mech_reg_coefs, predictive_errs_netmech, min_predictive_errs_netmech, mean_predictive_errs_netmech, double_std_predictive_errs_netmech

def initialise_adassp_reg_containers(epsilon_vec):
    adassp_reg_coef = {key: [] for key in epsilon_vec}
    predictive_err_adassp = {key: [] for key in epsilon_vec}
    min_predictive_err_adassp = {key: [] for key in epsilon_vec}
    mean_predictive_err_adassp = {key: [] for key in epsilon_vec}
    double_std_predictive_err_adassp = {key: [] for key in epsilon_vec}
    return adassp_reg_coef, predictive_err_adassp, min_predictive_err_adassp, mean_predictive_err_adassp, double_std_predictive_err_adassp