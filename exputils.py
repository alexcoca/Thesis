# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:48:59 2018

@author: alexc
"""

import numpy as np

def extract_data(results, multiple_datasets = False, max_dataset = 0, eps_dependent = False):
        
    max_utilities = []
    avg_samples_utility = []
    synthetic_datasets = []
   
    if not multiple_datasets:
        delta_cov_norms_f = []
        delta_corr_norms_2 = []
        avg_2_norms = []
        double_std_2_norms = []
        avg_f_norms = []
        double_std_f_norms = []
        sample_utilities = []
        max_sampled_utilities = []
        min_sampled_utilities = []
        double_std_utility = []
        avg_samples_score = []
        double_std_score = []
        delta_opt_avg = []
        delta_opt_best = []
        delta_opt_worst = []
        # test_set = []
        # private_data = []
        for result in results:
            delta_cov_norms_f.append(result['delta_cov_norms_f'])
            delta_corr_norms_2.append(result['delta_corr_norms_2'])
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
           # test_set.append(result['test_set'])
           # private_data.append(result['private_data'])
        test_set = results[0]['test_set']
        private_data = results[0]['private_data']    
        return delta_cov_norms_f, delta_corr_norms_2, avg_2_norms, double_std_2_norms, avg_f_norms, double_std_f_norms, max_utilities, sample_utilities, max_sampled_utilities, min_sampled_utilities, \
                avg_samples_utility, double_std_utility, avg_samples_score, double_std_score, synthetic_datasets, delta_opt_avg,\
                delta_opt_best, delta_opt_worst, test_set, private_data
    else:
        if not eps_dependent:
            # print (len(results))
            for result in results:
                max_utilities.append(result['max_utility'])
            test_set = results[0]['test_set']
            private_data = results[0]['private_data']
            return max_utilities, test_set, private_data
        else:
            # print (len(results))
            for result in results:
                avg_samples_utility.append(result['sample_utilities_avg'])
                synthetic_datasets.append(result['synthetic_data'])
            return avg_samples_utility, synthetic_datasets
                
def initialise_netmech_containers(epsilon_vec, multiple_datasets = False, max_dataset = 0):
    if not multiple_datasets:
        delta_cov_norms_f = {key: [] for key in epsilon_vec}
        delta_corr_norms_2 = {key: [] for key in epsilon_vec}
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
        return delta_cov_norms_f, delta_corr_norms_2, avg_2_norms, double_std_2_norms, avg_f_norms, double_std_f_norms, max_utilities, sample_utilities, max_sampled_utilities, min_sampled_utilities, \
                avg_samples_utility, double_std_utility, avg_samples_score, double_std_score, synthetic_datasets_vec, delta_opt_avg,\
                delta_opt_best, delta_opt_worst, test_set, private_data
    else:
        max_utilities = {key: [] for key in range(max_dataset)}
        avg_samples_utility = {key_outer: {key_inner : [] for key_inner in epsilon_vec}  for key_outer in range(max_dataset)}
        synthetic_datasets_vec  = {key_outer: {key_inner : [] for key_inner in epsilon_vec}  for key_outer in range(max_dataset)}
        test_set = {key: [] for key in range(max_dataset)}
        private_data = {key: [] for key in range(max_dataset)}
        return max_utilities, avg_samples_utility, synthetic_datasets_vec, test_set, private_data

def initialise_netmech_reg_containers(epsilon_vec, multiple_datasets = False, max_dataset = 0):
    if not multiple_datasets:
        net_mech_reg_coefs = {key: [] for key in epsilon_vec}
        predictive_errs_netmech = {key: [] for key in epsilon_vec}
        min_predictive_errs_netmech = {key: [] for key in epsilon_vec}
        mean_predictive_errs_netmech = {key: [] for key in epsilon_vec}
        double_std_predictive_errs_netmech = {key: [] for key in epsilon_vec}
        singular_indices = {key: [] for key in epsilon_vec}
        return net_mech_reg_coefs, predictive_errs_netmech, min_predictive_errs_netmech,\
                 mean_predictive_errs_netmech, double_std_predictive_errs_netmech, singular_indices
    else:
        net_mech_reg_coefs = {key_outer: {key_inner : [] for key_inner in epsilon_vec}  for key_outer in range(max_dataset)}
        predictive_errs_netmech = {key_outer: {key_inner : [] for key_inner in epsilon_vec}  for key_outer in range(max_dataset)}
        min_predictive_errs_netmech = {key_outer: {key_inner : [] for key_inner in epsilon_vec}  for key_outer in range(max_dataset)}
        mean_predictive_errs_netmech = {key_outer: {key_inner : [] for key_inner in epsilon_vec}  for key_outer in range(max_dataset)}
        double_std_predictive_errs_netmech = {key_outer: {key_inner : [] for key_inner in epsilon_vec}  for key_outer in range(max_dataset)}
        return net_mech_reg_coefs, predictive_errs_netmech, min_predictive_errs_netmech,\
         mean_predictive_errs_netmech, double_std_predictive_errs_netmech
        
def initialise_adassp_reg_containers(epsilon_vec, multiple_datasets = False, max_dataset = 0):
    if not multiple_datasets:
        adassp_reg_coef = {key: [] for key in epsilon_vec}
        predictive_err_adassp = {key: [] for key in epsilon_vec}
        min_predictive_err_adassp = {key: [] for key in epsilon_vec}
        mean_predictive_err_adassp = {key: [] for key in epsilon_vec}
        double_std_predictive_err_adassp = {key: [] for key in epsilon_vec}
        return adassp_reg_coef, predictive_err_adassp, min_predictive_err_adassp, mean_predictive_err_adassp, double_std_predictive_err_adassp
    else:
        adassp_reg_coef = {key_outer: {key_inner : [] for key_inner in epsilon_vec}  for key_outer in range(max_dataset)}
        predictive_err_adassp = {key_outer: {key_inner : [] for key_inner in epsilon_vec}  for key_outer in range(max_dataset)}
        min_predictive_err_adassp = {key_outer: {key_inner : [] for key_inner in epsilon_vec}  for key_outer in range(max_dataset)}
        mean_predictive_err_adassp = {key_outer: {key_inner : [] for key_inner in epsilon_vec}  for key_outer in range(max_dataset)}
        double_std_predictive_err_adassp = {key_outer: {key_inner : [] for key_inner in epsilon_vec}  for key_outer in range(max_dataset)}
        return adassp_reg_coef, predictive_err_adassp, min_predictive_err_adassp, mean_predictive_err_adassp, double_std_predictive_err_adassp

def get_expected_statistics(avg_samples_utility, mean_predictive_errs_netmech, double_std_predictive_errs_netmech,\
                            mean_predictive_err_adassp, double_std_predictive_err_adassp, min_predictive_errs_netmech, \
                            min_predictive_err_adassp, epsilon_vec):
    ''' This helper function calculates mean/max/min of various quantities (e.g. avg_samples_utility) across datasets, for
    various epsilon and lattice density combinations'''
    def get_helper(obj, epsilon_vec):
        helper = {key: [] for key in epsilon_vec}
        for dataset_number in obj.keys():
            for epsilon in obj[dataset_number].keys():
                helper[epsilon].append(obj[dataset_number][epsilon])
        return helper
    
    def get_stats(helper, epsilon_vec, kinds = ['mean']):
        statistics = [{key: [] for key in epsilon_vec} for _ in range(len(kinds))]
        for index in range(len(kinds)):
            if kinds[index] == 'mean':
                for epsilon in helper.keys():
                    statistics[index][epsilon] = np.mean(helper[epsilon], axis = 0)
            if kinds[index] == 'min':
                for epsilon in helper.keys():
                    statistics[index][epsilon] = np.min(helper[epsilon], axis = 0)
            if kinds[index] == 'max':
                for epsilon in helper.keys():
                    statistics[index][epsilon] = np.max(helper[epsilon], axis = 0)
        if len(kinds) > 1:
            return tuple(statistics)
        else: 
            return statistics[0]
    
    expected_avg_utility, min_avg_utility, max_avg_utility = get_stats(get_helper(avg_samples_utility, epsilon_vec),\
                                                                       epsilon_vec, kinds = ['mean','min', 'max'])
    expected_mean_predictive_errs_netmech =  get_stats(get_helper(mean_predictive_errs_netmech, epsilon_vec), epsilon_vec, kinds = ['mean'])
    expected_double_std_predictive_errs_netmech = get_stats(get_helper(double_std_predictive_errs_netmech, epsilon_vec), epsilon_vec, kinds = ['mean'])
    expected_mean_predictive_err_adassp =  get_stats(get_helper(mean_predictive_err_adassp, epsilon_vec), epsilon_vec,  kinds = ['mean'])
    expected_double_std_predictive_err_adassp = get_stats(get_helper(double_std_predictive_err_adassp, epsilon_vec), epsilon_vec, kinds = ['mean'])
    expected_min_predictive_errs_netmech, min_min_predictive_errs_netmech, max_min_predictive_errs_netmech = get_stats(get_helper(min_predictive_errs_netmech, epsilon_vec), epsilon_vec, kinds = ['mean','min', 'max'])
    expected_min_predictive_err_adassp, min_min_predictive_err_adassp, max_min_predictive_err_adassp = get_stats(get_helper(min_predictive_err_adassp, epsilon_vec), epsilon_vec, kinds = ['mean','min', 'max'])

    return  expected_avg_utility, min_avg_utility, max_avg_utility, expected_mean_predictive_errs_netmech,\
            expected_double_std_predictive_errs_netmech, expected_mean_predictive_err_adassp,\
            expected_double_std_predictive_err_adassp, expected_min_predictive_errs_netmech, min_min_predictive_errs_netmech,\
            max_min_predictive_errs_netmech, expected_min_predictive_err_adassp, min_min_predictive_err_adassp,\
            max_min_predictive_err_adassp

def get_optimal_utilities_statistics(max_utilities):
    ''' This function calculates the expectiation/min/max of the maximum utility across datasets '''
    expected_optimal_utilities = np.mean(list(max_utilities.values()), axis = 0)
    min_optimal_utilities = np.min(list(max_utilities.values()), axis = 0)
    max_optimal_utilities = np.max(list(max_utilities.values()), axis = 0)
    return expected_optimal_utilities, min_optimal_utilities, max_optimal_utilities