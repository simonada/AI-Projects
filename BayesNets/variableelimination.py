import structurelearning
import pandas as pd
import math
from operator import mul
from functools import reduce
import itertools
from collections import defaultdict
import numpy as np
from collections import namedtuple


def variableElimination(index, observations, model):
    #print('Starting Variable Elimination!')
    adj_matrix, cpt_list = model
    if not observations[0]:
        return cpt_list[index]

    observed_indices = [o_id for o_id, o_value in observations]

    # Local CPT instantiated by evidence
    factors_cpts = factorize(adj_matrix, observations, cpt_list)
    reduced_factors = factors_cpts.copy()

    # Eliminate irrelevant variables, i.e. those which are not successors to any of the observed or queried variable
    indices_to_keep = get_all_ancestors(index, observed_indices, adj_matrix)
    # Eliminate all hidden variables (i.e. except target and observed)
    for hidden_var in range(len(cpt_list)):
        if hidden_var not in indices_to_keep:
            # ignore all irrelevant variable
            for key, value in reduced_factors.items():
                if key.var == hidden_var:
                    del reduced_factors[key]
                    break
            continue
        if hidden_var != index and hidden_var not in observed_indices:
            reduced_factors = eliminate_variable(hidden_var, reduced_factors)
    # Join all remaining factors
    if len(reduced_factors) != 1:
        #print('Only observed and target left:', reduced_factors)
        reduced_factors, reduced_cpt = pointwise_product(index, reduced_factors, reduced_factors.copy())

    #print('Variable elimination finished!')
    final_cpt = normalize(reduced_cpt)
    # print('Final cpt:', final_cpt)
    # print('Final factors:', reduced_factors)

    return final_cpt

def get_all_ancestors(index, observed_indices, adj_matrix):
    all_present_vars = observed_indices
    all_present_vars.append(index)
    ancestors = []
    adj_matrix_transposed = pd.DataFrame(adj_matrix).T

    for var_id, parents in adj_matrix_transposed.iterrows():
        if var_id in all_present_vars:
            parents_var_ids = [i for i, e in enumerate(list(parents)) if e == 1]
            if parents_var_ids:
                ancestors += parents_var_ids

    all_present_vars = all_present_vars + ancestors
    #flat_list = [item for sublist in all_present_vars for item in sublist]
    return all_present_vars


def normalize(factor):
    return factor / np.sum(factor)


def eliminate_variable(var, all_factors_dict):
    print()
    #print('Eliminating variable', var)
    Factor = namedtuple("Factor", ["var", "conditions"])
    new_factors_dict = all_factors_dict.copy()

    # get the factors which will be multiplied pointwise
    # prepare to update the factors dict based on the joined factors
    relevant_factors, new_factors_dict = get_relevant_and_reduced_factors(var, all_factors_dict)
    if len(relevant_factors) == 1:
        #print('No factors to join, variable has no influence on the query.')
        # print(relevant_factors)
        del all_factors_dict[list(relevant_factors.keys())[0]]

   # print('Relevant Factors dictionary contains:', relevant_factors.keys())
   # print('Joining factors', relevant_factors.keys())
    intermediate_factor, intermediate_cpt = pointwise_product(var, relevant_factors, all_factors_dict)

    # if no factors were left out, we're ready to sum out
    # sum out the variable by summing across the arrays in the cpt list
    # each summed value represents the probability for the possible values of the outter most variable
    # print('Before sumout interm factor', intermediate_factor, 'interm cpt', intermediate_cpt)
    reduced_factor, reduced_cpt = sum_out(var, intermediate_factor, intermediate_cpt)
    new_factors_dict[reduced_factor] = reduced_cpt

    return new_factors_dict


def pointwise_product(var, relevant_factors, all_factors_dict):
    Factor = namedtuple("Factor", ["var", "conditions"])
    processed_factors = relevant_factors.copy()
    # loop over the relevant factors and multiply pointwise until a single factor is left
    # intermediate cpt for transitioning between multiple joins
    intermediate_factor = Factor(var=1, conditions=(0))
    intermediate_cpt = np.array([])
    for factor_1, factor_2 in group_factors(relevant_factors, 2):
        #print('Factors to join',factor_1, factor_2)
        # since we're grouping pairwise, we want to make sure there is no factor left out
        # in case of an odd number of factors
        del processed_factors[factor_1]
        del processed_factors[factor_2]
        # Here in join factors is where the pointwise product takes place
        joined_factor, cpt_multiplied = join_factors(var, factor_1, factor_2, all_factors_dict)
        if intermediate_cpt.size == 0:
            intermediate_cpt = cpt_multiplied
            intermediate_factor = joined_factor
            all_factors_dict[intermediate_factor] = intermediate_cpt
        else:
            # pointwise product with the previously joined factors, thus we accumulate the results
            intermediate_factor, intermediate_cpt = join_factors(var, intermediate_factor, joined_factor, all_factors_dict)
            all_factors_dict[intermediate_factor] = intermediate_cpt
    if processed_factors:
        # there is one more left as we had odd number to join
        factor_left = list(processed_factors.keys())[0]
        intermediate_factor, intermediate_cpt = join_factors(factor_left.var, intermediate_factor, factor_left, all_factors_dict)

    return intermediate_factor, intermediate_cpt


def join_factors(var, factor_1, factor_2, factors_dict):
    #print('Joining for var', var)
    #print('Joining', factor_1, factor_2)
    Factor = namedtuple("Factor", ["var", "conditions"])
    # Build new factor name = merge of the involved variable names
    factor_1_vars = get_involved_factor_variables(factor_1)
    factor_2_vars = get_involved_factor_variables(factor_2)
    factor_var = tuple(set(factor_1_vars + factor_2_vars))

    if var in factor_1_vars:
        position_of_target_f1 = list(factor_1_vars).index(var)
    else:
        position_of_target_f1 = -1
    if var in factor_2_vars:
        position_of_target_f2 = list(factor_2_vars).index(var)
    else:
        position_of_target_f2 = -1

    # pointwise multiplication based on several possible cases dependent on the dimensions
    # print('Pointwise multiplication for', factor_1, factor_2)
    #print('Factors dict', factors_dict)
    # print('CPT_1: ', factors_dict[factor_1].shape)
    # print(factors_dict[factor_1])
    # print('CPT_2: ', factors_dict[factor_2].shape)
    # print(factors_dict[factor_2])

    cpt_factor_1 = factors_dict[factor_1]
    cpt_factor_2 = factors_dict[factor_2]
    cpt_factor_1_shape = cpt_factor_1.shape
    cpt_factor_2_shape = cpt_factor_2.shape
    #print('CPTs to multiply:', cpt_factor_1, cpt_factor_2)
    #print(cpt_factor_1_shape, cpt_factor_2_shape)
    cpt_multiplied = []
    # case example (3,) * (2, 3, 2) => (1,3,1) * (2,3,2)
    if (cpt_factor_1_shape == cpt_factor_2_shape):
        cpt_multiplied = cpt_factor_1 * cpt_factor_2

    elif (len(cpt_factor_1_shape) == 1 and len(cpt_factor_2_shape) == 3) or (
            len(cpt_factor_2_shape) == 1 and len(cpt_factor_1_shape) == 3):
        if len(cpt_factor_1_shape) == 1:
            if cpt_factor_1_shape[0] == cpt_factor_2_shape[1]:
                cpt_multiplied = multiply_with_padding_on_both_sides(cpt_factor_1, cpt_factor_2)
        else:
            if cpt_factor_2_shape[0] == cpt_factor_1_shape[1]:
                cpt_multiplied = multiply_with_padding_on_both_sides(cpt_factor_2, cpt_factor_1)
    # case shared dimension is at the same position e.g. (R,T) * (L,T) = > (R,T,L), where shared is at position 1
    elif (len(cpt_factor_1_shape) == len(cpt_factor_2_shape) and (position_of_target_f1 == position_of_target_f2)):
        shared_dim = position_of_target_f1
        cpt_multiplied = multiply_pintwise_across_shared_dim(cpt_factor_1, cpt_factor_2, shared_dim)
    elif (cpt_factor_1_shape != cpt_factor_2_shape):
        # case that one of them is a plain number
        if not cpt_factor_1_shape or not cpt_factor_2_shape:
            cpt_multiplied = cpt_factor_1 * cpt_factor_2
        # otherwise multiply where the target is with the other probabilities
        elif var in factor_1.var:
            #print(cpt_factor_1)
            #print(cpt_factor_2)
            cpt_multiplied = []
            for val in cpt_factor_1:
                partial_res = []
                for val_2 in cpt_factor_2:
                    partial_res.append(val * val_2)
                cpt_multiplied.append(sum(partial_res))
    else:
        cpt_multiplied = cpt_factor_1 * cpt_factor_2



    joined_factor = Factor(var=factor_var, conditions=())
    factors_dict[joined_factor] = cpt_multiplied
    #print('Joined Factor:', joined_factor)
    #print('Joined CPT', cpt_multiplied)
    #print()
    return joined_factor, cpt_multiplied


def multiply_pintwise_across_shared_dim(cpt_1, cpt_2, dim):
    # print('original dimensions', cpt_1.shape, cpt_2.shape)
    # print('shared dimension', dim)
    cpt_new_dimensions = merge_dimensions(cpt_1, cpt_2, dim)
    cpt_new_array = np.zeros(cpt_new_dimensions, dtype=np.float64)
    # print('new joined cpt shape', cpt_new_array.shape)
    dim_list = list(cpt_new_dimensions)
    combinations_first_factor = get_values_combinations(cpt_1.shape)
    # print('combinations to loop over', combinations_first_factor)
    for combination in combinations_first_factor:
        array_indices = list(combination)
        shared_dim_value = array_indices[dim]
        firs_factor_values = cpt_1[combination]
        second_factor_conditioned_values = condition_cpt_no_par(cpt_2, dim, shared_dim_value)
        product = firs_factor_values * second_factor_conditioned_values
        cpt_new_array[combination] = product
    return cpt_new_array


def get_values_combinations(cpt):
    list_possible_values = []
    for dim in cpt:
        list_possible_values.append(list(range(dim)))
    tuples_combinations = []
    for element in itertools.product(*list_possible_values):
        tuples_combinations.append(element)
    return tuples_combinations


def condition_cpt_no_par(old_cpt, par_id, par_value):
    # print(old_cpt)
    # print(old_cpt.shape)
    cpt_dimensions = old_cpt.shape
    s = slice(None)
    conditions = []
    # first entry is the variable itself, we always keep it (assume that the queries target is never conditioned)
    # par_id += 1  # since they start from 0, but the 0 index is alway the variable itself
    for dim in range(len(cpt_dimensions)):
        if dim == par_id:
            conditions.append(par_value)
        else:
            conditions.append(s)
    # print(conditions)
    conditioned_cpt = old_cpt[tuple(conditions)]
    return conditioned_cpt


def merge_dimensions(cpt_1, cpt_2, dim):
    cpt_new_dimensions = []
    for e in range(len(cpt_1.shape)):
        if e != dim:
            cpt_new_dimensions.append(cpt_1.shape[e])
    for e in range(len(cpt_2.shape)):
        if e != dim:
            cpt_new_dimensions.append(cpt_2.shape[e])
    cpt_new_dimensions.insert(dim, list(cpt_1.shape)[dim])
    cpt_new_dimensions = tuple(cpt_new_dimensions)
    return cpt_new_dimensions


def multiply_with_padding_on_both_sides(factor_cpt_to_pad, factor_to_multiply):
    ar_pad_right = factor_cpt_to_pad[:, np.newaxis]
    ar_pad_right_left = ar_pad_right[np.newaxis, :]
    return ar_pad_right_left * factor_to_multiply


def get_involved_factor_variables(factor):
    involved_var = ()
    if isinstance(factor.var, int):
        involved_var += (factor.var,)
    else:
        involved_var += factor.var
    if isinstance(factor.conditions, int):
        involved_var += (factor.conditions,)
    else:
        involved_var += factor.conditions
    return involved_var


def sum_out(var_id, factor, cpt):
    # print('Summing out', var_id, 'for factor,', factor, 'and cpt', cpt)
    Factor = namedtuple("Factor", ["var", "conditions"])

    if len(cpt.shape) == 1:
        summed_cpt = sum(cpt)
    else:
        summed_cpt = [sum(x) for x in cpt]

    factor_var = get_involved_factor_variables(factor)
    factor_var = tuple(var for var in factor_var if var != var_id)

    return Factor(var=factor_var, conditions=()), summed_cpt


# return 2 data structures: the relevant for further processing factors,
# and the dictionary without the factors that will be merged
def get_relevant_and_reduced_factors(var, all_factors_dict):
    relevant_factors = all_factors_dict.copy()
    reduced_factors = all_factors_dict.copy()

    for factor in all_factors_dict.keys():
        if not check_if_var_in_factor(var, factor):
            del relevant_factors[factor]
        else:
            del reduced_factors[factor]

    return relevant_factors, reduced_factors


def check_if_var_in_factor(var_id, factor):
    if isinstance(factor.conditions, int):
        factor_var = (factor.var,) + (factor.conditions,)
    else:
        if isinstance(factor.var, int):
            factor_var = (factor.var,) + factor.conditions
        else:
            factor_var = factor.var + factor.conditions
    if var_id in factor_var:
        return True
    else:
        return False


def group_factors(rel_factors, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(rel_factors)] * n)


def factorize(adj_matrix, observations, cpt_list):
    """
    Returns a factors dictionary, where (key = named tuple, i.e. var_id, and array of var_ids on which it's conditioned,
    values = cpt list for the arrangement)
    """
    Factor = namedtuple("Factor", ["var", "conditions"])
    # view of the data as (rows = array of the variable parents)
    adj_matrix_transposed = pd.DataFrame(adj_matrix).T
    observed_indices = [o_id for o_id, o_value in observations]
    factors_cpts = {}
    for var_id, parents in adj_matrix_transposed.iterrows():
        parents_var_ids = [i for i, e in enumerate(list(parents)) if e == 1]
        factor = Factor(var=var_id, conditions=tuple(parents_var_ids))
        # each factor is associated with the correct cpt
        unconditioned_cpt = cpt_list[var_id]
        # check if there are parents which are observed, i.e. intersection of observed and parents ids
        observed_parents = list(set(parents_var_ids) & set(observed_indices))
        if observed_parents:
            # compute conditioned cpt for the observed parents
            for par_id in observed_parents:
                # get the relevant index as it is from dependent variable perspective
                par_rel_id = parents_var_ids.index(par_id)
                par_value = [o_value for o_id, o_value in observations if o_id == par_id][0]
                conditioned_cpt = condition_cpt(unconditioned_cpt, par_rel_id, par_value)
                factors_cpts[factor] = conditioned_cpt
        elif var_id in observed_indices:
            var_value = [o_value for o_id, o_value in observations if o_id == var_id][0]
            conditioned_cpt = unconditioned_cpt[var_value]
            factors_cpts[factor] = conditioned_cpt
        else:
            factors_cpts[factor] = unconditioned_cpt
    return factors_cpts


def condition_cpt(old_cpt, par_id, par_value):
    # print(old_cpt)
    # print(old_cpt.shape)
    cpt_dimensions = old_cpt.shape
    s = slice(None)
    conditions = []
    # first entry is the variable itself, we always keep it (assume that the queries target is never conditioned)
    par_id += 1  # since they start from 0, but the 0 index is alway the variable itself
    for dim in range(len(cpt_dimensions)):
        if dim == par_id:
            conditions.append(par_value)
        else:
            conditions.append(s)
    conditioned_cpt = old_cpt[tuple(conditions)]
    return conditioned_cpt


if __name__ == '__main__':
    matrix = [[1, 1, 0, 1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 1, 1, 0, 1, 0]]
    total_score, adj_matrix = structurelearning.K2Algorithm(2, matrix, structurelearning.K2Score)
    print()
    # cpt = MLEstimationVariable(2, [0, 1], matrix)
    cptList = structurelearning.MLEstimatorVariable(adj_matrix, matrix)

    print()
    print('CPT List:')
    print(cptList)
    print()
    model = (adj_matrix, cptList)
    print()
    reduced_factors = variableElimination(2, [(1, 0)], model)

    print()
    print('Elimination results: ')
    print(reduced_factors)
