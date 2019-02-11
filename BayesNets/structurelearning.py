import pandas as pd
import math
from operator import mul
from functools import reduce
import itertools
from collections import defaultdict
import numpy as np


def K2Algorithm(K, data, scoreFunction):
    verbose = False
    # NB: if we want to test with the example from the lecture the dimensions are handled differently, use:
    n = len(data)

    # the number of variables, i.e. the number of dimensions for the evaluation datasets for Part II
    #n = len(data[0])
    #print('data dimensions', n)
    structure_dict = defaultdict(list)
    adj_matrix = np.zeros((n, n), dtype=np.int32)
    total_score = 0
    for i in range(n):
        parents = []
        p_old = scoreFunction(i, parents, data)
        proceed = True
        # predec are all variables before that index
        predecessors = list(range(i))
        if verbose:
            print()
            print('New node!', i)
            print("P_old:", p_old)
            print('Predecessors', predecessors)
        while proceed and len(parents) < K:
            if predecessors:
                # pick best predecessor, evaluate isolated impact
                separate_impact = []
                for pred in predecessors:
                    parents_temp = parents.copy()
                    parents_temp.append(pred)
                    separate_impact.append(scoreFunction(i, parents_temp, data))
                # print('separate impact', separate_impact)
                p_new = max(separate_impact)
                max_parent_index = separate_impact.index(max(separate_impact))
                # print('max index', max_parent_index, 'with score', p_new)
                if (p_new > p_old):
                    p_old = p_new
                    parent_id = predecessors.pop(max_parent_index)
                    structure_dict[i].append(parent_id)
                    parents.append(parent_id)
                    total_score += p_new
                else:
                    #print('No improvement possible.')
                    proceed = False

            else:
                print('No predecessors available.')
                proceed = False

    for key, values in structure_dict.items():
        for parent in values:
            adj_matrix[parent][key] = 1

    print()
    print('Structure Learned!')
    print(structure_dict)
    #print('Total Score:', total_score)
    #print(adj_matrix)
    return total_score, adj_matrix


def K2ScoreLogs(variable_index, parents, data):
    variable = data[variable_index]
    #print('INPUT to K2Score', variable, parents)
    # print(variable)
    pd_matrix = pd.DataFrame(data)
    # print(pd_matrix)
    r_i = len(set(variable))

    if not parents:
        # work with the single variable array, no need to condition
        N_i_j = len(variable)
        result_factorials_sums = get_log_sum_factorials(variable)
        #print(r_i, N_i_j)
        #print((math.factorial(r_i - 1)) / (math.factorial(N_i_j + r_i - 1)))
        log_equation_part = math.log((math.factorial(r_i - 1)), 2) - math.log((math.factorial(N_i_j + r_i - 1)), 2)
        return log_equation_part + result_factorials_sums
    else:
        # print(parents, 'parents')
        cart_product_parents = get_values_combinations(parents, data)
        partial_sums = []
        # for each combination, loop over all values of the parents which are part of the combination
        for comb in cart_product_parents:
            pd_matrix_conditioned = pd_matrix.copy().T

            for i in range(len(comb)):
                combination_elements = comb[i]
                par_index, par_value = combination_elements
                # iteratively reduce the df based on the parent values of interest
                pd_matrix_conditioned = pd_matrix_conditioned.loc[pd_matrix_conditioned.iloc[:, par_index] == par_value]
            # print(comb, 'conditioned matrix:', pd_matrix_conditioned)
            # print('relevant indices', list(pd_matrix_conditioned.T))
            indices_to_keep = list(pd_matrix_conditioned.T)
            variable_conditioned = [variable[i] for i in indices_to_keep]
            # get the values of the target variable under this conditions
            # variable_conditioned = list(pd_matrix_conditioned.T.iloc[var_id])
            # print('conditioned target:', variable_conditioned)
            result_factorials_sum = get_log_sum_factorials(variable_conditioned)
            # Sum of occurences N_i_j_k
            N_i_j = len(variable_conditioned)

            # sum here, take log of the first product, the log for the second one is taken care of above
            log_equation_part = math.log((math.factorial(r_i - 1)), 2) - math.log((math.factorial(N_i_j + r_i - 1)), 2)
            partial_sums.append(log_equation_part + result_factorials_sum)

        final_sum = reduce((lambda x, y: x + y), partial_sums)
        #print('FINAL score', final_sum)
        return final_sum


def get_log_sum_factorials(variable):
    # case when no values under the condition are available
    if not variable:
        return 1
    # i.e. sum over the log of the factorials
    values_frequencies = [(i, variable.count(i)) for i in set(variable)]
    # print('L_i_j_k (value,frequency)', values_frequencies)
    # unique values that the variable can take
    values = [i[0] for i in values_frequencies]
    r_i = len(values)
    # frequency list for the unique values that the variable can take
    frequencies = [i[1] for i in values_frequencies]

    # log here
    result_factorials = map(math.factorial, frequencies)
    result_factorials = [math.log(y, 2) for y in result_factorials]

    #  sum here
    result_factorials_product = reduce((lambda x, y: x + y), result_factorials)
    return result_factorials_product


def K2Score(variable_index, parents, data):
    variable = data[variable_index]
    #print('INPUT to K2Score', variable, parents)
    # print(variable)
    pd_matrix = pd.DataFrame(data)
    # print(pd_matrix)
    r_i = len(set(variable))

    if not parents:
        # work with the single variable array, no need to condition
        N_i_j = len(variable)
        result_factorials_product = get_product_factorials(variable)
        return (math.factorial(r_i - 1) / math.factorial(N_i_j + r_i - 1)) * result_factorials_product
    else:
        # print(parents, 'parents')
        cart_product_parents = get_values_combinations(parents, data)
        partial_products = []
        # for each combination, loop over all values of the parents which are part of the combination
        for comb in cart_product_parents:
            pd_matrix_conditioned = pd_matrix.copy().T

            for i in range(len(comb)):
                combination_elements = comb[i]
                par_index, par_value = combination_elements
                # iteratively reduce the df based on the parent values of interest
                pd_matrix_conditioned = pd_matrix_conditioned.loc[pd_matrix_conditioned.iloc[:, par_index] == par_value]
            # print(comb, 'conditioned matrix:', pd_matrix_conditioned)
            # print('relevant indices', list(pd_matrix_conditioned.T))
            indices_to_keep = list(pd_matrix_conditioned.T)
            variable_conditioned = [variable[i] for i in indices_to_keep]
            # get the values of the target variable under this conditions
            # variable_conditioned = list(pd_matrix_conditioned.T.iloc[var_id])
            # print('conditioned target:', variable_conditioned)
            result_factorials_product = get_product_factorials(variable_conditioned)
            # Sum of occurences N_i_j_k
            N_i_j = len(variable_conditioned)
            partial_products.append(
                (math.factorial(r_i - 1) / math.factorial(N_i_j + r_i - 1)) * result_factorials_product)
        final_product = reduce((lambda x, y: x * y), partial_products)
        # print('Separate scores before multiplication', partial_products)
        print('FINAL score', final_product)
        return final_product


def get_product_factorials(variable):
    # case when no values under the condition are available
    if not variable:
        return 1
    values_frequencies = [(i, variable.count(i)) for i in set(variable)]
    # print('L_i_j_k (value,frequency)', values_frequencies)
    # unique values that the variable can take
    values = [i[0] for i in values_frequencies]
    r_i = len(values)
    # frequency list for the unique values that the variable can take
    frequencies = [i[1] for i in values_frequencies]
    result_factorials = map(math.factorial, frequencies)
    result_factorials_product = reduce((lambda x, y: x * y), result_factorials)
    return result_factorials_product


def BICScore(variable_index, parents, data):
    variable = data[variable_index]
    #print('INPUT to BICScore', variable, parents)
    # print(variable)
    pd_matrix = pd.DataFrame(data)
    # print(pd_matrix)
    r_i = len(set(variable))
    N = len(data)

    if not parents:
        # work with the single variable array, no need to condition
        N_i_j = len(variable)
        result_bic_sum = 2 * get_bic_sum(variable, N_i_j, 0, N)
        return result_bic_sum
    else:
        #print(parents, 'parents')
        cart_product_parents = get_values_combinations(parents, data)
        partial_sums = []
        # for each combination, loop over all values of the parents which are part of the combination
        for comb in cart_product_parents:
            pd_matrix_conditioned = pd_matrix.copy().T

            for i in range(len(comb)):
                combination_elements = comb[i]
                par_index, par_value = combination_elements
                # iteratively reduce the df based on the parent values of interest
                pd_matrix_conditioned = pd_matrix_conditioned.loc[pd_matrix_conditioned.iloc[:, par_index] == par_value]
            #print(comb, 'conditioned matrix:', pd_matrix_conditioned)
            #print('relevant indices', list(pd_matrix_conditioned.T))
            indices_to_keep = list(pd_matrix_conditioned.T)
            variable_conditioned = [variable[i] for i in indices_to_keep]
            # get the values of the target variable under this conditions
            # variable_conditioned = list(pd_matrix_conditioned.T.iloc[var_id])
            #print('conditioned target:', variable_conditioned)
            # Sum of occurences N_i_j_k
            N_i_j = len(variable_conditioned)
            result_bic_sum = get_bic_sum(variable_conditioned, N_i_j, len(comb), N)
            partial_sums.append(result_bic_sum)

        # here the outer summations are performed, i.e. for all q_i
        final_sum = 2 * sum(partial_sums)
        #print('Separate scores before summation', partial_sums)
        #print('FINAL score', final_sum)
        return final_sum


def get_bic_sum(variable, n_i_j, q_i, N):
       # here the inner summations are performed
    values_frequencies = [(i, variable.count(i)) for i in set(variable)]
    #print('L_i_j_k (value,frequency)', values_frequencies)
    # unique values that the variable can take
    values = [i[0] for i in values_frequencies]
    r_i = len(values)
    # frequency list for the unique values that the variable can take, i.e. L_i_j_k
    frequencies = [i[1] for i in values_frequencies]
    summations = []
    # N = Number of training instances
    for n in frequencies:
        partial_sum = n * math.log(n / n_i_j, 2) - q_i * (r_i - 1) * math.log(N, 2)
        summations.append(partial_sum)

    return sum(summations)


def MLEstimatorVariable(graph, data):
    cptList = []
    for var_index in range(len(data)):
        # get parents, i.e. the Index of each parent in the graph, where there is a 1 in the position of the variable
        parents = [i for i, adj_array in enumerate(graph) if adj_array[var_index] == 1]
        cpt = MLEstimationVariable(var_index, parents, data)
        cptList.append(cpt)
    return cptList


def MLEstimationVariable(variable_index, parents, data):
    variable = data[variable_index]

    # NOTE: we need to have a mapping between the cpt indices and the variable indices, so that later we can update the cpt values in the right order
    # Have a dictionary where for each variable: key= var_id, values = list of unique values the var can take
    # On query time look up the values id, get the list and from that list the index of the value, based on which the cpt will be updated
    mapping_dict = defaultdict(list)
    variable_values_frequencies = [(i, variable.count(i)) for i in set(variable)]

    # IDEA:
    # 1. Create the cpt based on the dimensions of the input.
    # The dimensions are given by the number of unique values that each variable can take.
    dimensions = (len(set(variable)),)
    mapping_dict[variable_index] = list(set(variable))

    if not parents:
        cpt_array = np.zeros(dimensions, dtype=np.float64)
        total_obs = len(variable)
        i = 0
        for var_value in set(variable):
            value_frq = [v for i, v in variable_values_frequencies if i == var_value]
            avg_frq = value_frq[0] / total_obs
            cpt_array[i] = avg_frq
            i += 1
        return cpt_array

    for p_index in parents:
        # get the number of values each parent can have
        unique_values = [(i) for i in set(data[p_index])]
        mapping_dict[p_index] = list(unique_values)
        dimensions += (len(unique_values),)
    #print(dimensions)
    cpt_array = np.zeros(dimensions, dtype=np.float64)
    #print(cpt_array)
    # 2. Get all possible combinations between the values of all variables
    all_var_indices = [variable_index] + parents
    #print(all_var_indices)
    all_combinations = get_values_combinations(all_var_indices, data)
    #print(all_combinations)
    #print(mapping_dict)
    # 3. For each combination condition the table and get the probability for the case
    get_conditioned_frequency(all_combinations[0], data)
    for combination in all_combinations:
        # get total frequency of the target variable for that value before conditioning
        # the target variable is the first element
        target_id, target_value = combination[0]
        total_frq = [v for i, v in variable_values_frequencies if i == target_value]
        # get conditioned frequency of the value
        conditioned_frq = get_conditioned_frequency(combination, data)
        # compute probability
        #print(conditioned_frq, total_frq)
        probability = round(conditioned_frq / total_frq[0], 3)
        #print('Probability:', probability)
        # update the relevant position in the cpt, first get the list of indices, then update the cpt
        combination_indices = []
        for var_index, var_value in combination:
            # index of the value
            value_index = mapping_dict[var_index].index(var_value)
            combination_indices.append([value_index])
        # print('CPT to update:', combination_indices)
        # the tuple is used to access the relevant dimension in the cpt array
        cpt_array[tuple(combination_indices)] = probability

    #print('Final CPT', cpt_array)
    return cpt_array


def get_conditioned_frequency(combinations_tuple, data):
    pd_matrix = pd.DataFrame(data)
    pd_matrix_conditioned = pd_matrix.copy().T
    for i in range(len(combinations_tuple)):
        combination_elements = combinations_tuple[i]
        par_index, par_value = combination_elements
        # iteratively reduce the df based on the parent values of interest
        pd_matrix_conditioned = pd_matrix_conditioned.loc[pd_matrix_conditioned.iloc[:, par_index] == par_value]
    # print(combinations_tuple, 'conditioned matrix:', pd_matrix_conditioned)
    remaining_indices = list(pd_matrix_conditioned.T)
    # print(len(remaining_indices))
    return len(remaining_indices)


def get_values_combinations(node_indices, data):
    values_all_parents = []
    for p_index in node_indices:
        values_frequencies = [(i, data[p_index].count(i)) for i in set(data[p_index])]
        values = [i[0] for i in values_frequencies]
        values_all_parents.append(values)
    return get_cart_product(values_all_parents, node_indices)


def get_cart_product(lists, parents):
    combinations = []

    for element in itertools.product(*lists):
        combination = []
        for p in range(len(parents)):
            tup = (parents[p], element[p])
            combination.append(tup)
        # in order to be able to access the parent ids later, the data structure for a single combination is a list of tuples, where tuple
        # is defined for each parent in the combination as (parent_id, parent_value)
        combinations.append(combination)
    #print('Cartesian product combinations (parent_id, value)', combinations)
    return combinations


if __name__ == '__main__':
    data = [[1, 1, 0, 1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 1, 1, 0, 1, 0]]

    print('= Structure Learning =')
    print('--- K2Score ---')
    total_score, adj_matrix = K2Algorithm(2, data, K2Score)
    print(adj_matrix)
    print()
    print('--- K2Score Logs---')
    total_score, adj_matrix = K2Algorithm(2, data, K2ScoreLogs)
    print(adj_matrix)
    print()
    print('--- BIC Score ---')
    total_score, adj_matrix = K2Algorithm(2, data, BICScore)
    print(adj_matrix)
    print()

    print('= Parameter Learning =')
    cptList = MLEstimatorVariable(adj_matrix, data)
    print(cptList)
