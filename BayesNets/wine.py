import structurelearning
import csv
import numpy as np
import random
import time
import timeutils
import variableelimination
import pickle
import signal


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def read_data(file_name):
    data = []
    file_path = 'data/' + file_name

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def get_train_test_split(data, p):
    # p should be the fraction, e.g. 0.7
    test_p = int(len(data) * p)
    # lists = random.sample(data, len(data) * p)
    random.shuffle(data)
    train_data = data[:test_p]
    test_data = data[test_p:]
    return train_data, test_data


def learn_structure(K, train_sample, scoreFunction):
    start = time.time()
    try:
        with timeout(seconds=1800):
            total_score, adj_matrix = structurelearning.K2Algorithm(K, train_sample, scoreFunction)
            end = time.time()
            timeutils.print_time_output(start, end)
            print()
            print('Total Score with K=', K)
            print(total_score)
            print()
            # print('Adjacency matrix with K =', K)
            # print(adj_matrix)

    except Exception as ex:
        if isinstance(ex, TimeoutError):
            print('Algorithm took more than 30min. Interrupting...')
            end = time.time()
            timeutils.print_time_output(start, end)
        else:
            print("Error!", ex.with_traceback())
        # break
    return total_score, adj_matrix


def learn_parameters(adj_matrix, train_sample):
    start = time.time()
    try:
        with timeout(seconds=1800):
            cptList = structurelearning.MLEstimatorVariable(adj_matrix, train_sample)
            end = time.time()
            timeutils.print_time_output(start, end)
    except Exception as ex:
        if isinstance(ex, TimeoutError):
            print('Algorithm took more than 30min. Interrupting...')
            end = time.time()
            timeutils.print_time_output(start, end)
        else:
            print("Error!", ex.with_traceback())
        # break
    return cptList


def inference_evaluation(test_data, model):
    print('todo')
    d = len(test_data)
    #
    accuracy = 0
    # get all possible values for the target, i.e. map the predicted to the true one
    for obs in test_data:
        conditions = []
        # last variable is the one of interest
        target_class = obs[-1]
        for var in range(len(obs) - 1):
            # condition should be (var_id, observed_value)
            var_condition = (var, obs[var])
            conditions.append(var_condition)
        probabilities = variableelimination.variableElimination(11, conditions, model)
        max_probability_class = probabilities.index(max(probabilities))
        if max_probability_class == target_class:
            accuracy += 1
    return accuracy / d


def test_p(all_data, scoreFunction):
    p = 0.2

    while p <= 1:
        print('Running with p', p)
        train_sample, test_sample = get_train_test_split(all_data, p)
        print('Train and test dataset size: ', len(train_sample), ',', len(test_sample))
        train_sample = list(map(list, zip(*train_sample)))

        print('---- Structure Learning ----')
        score, adj_matrix = learn_structure(2, train_sample, scoreFunction)
        adj_matrix_np = np.array(adj_matrix)
        avg_nr_parents = sum(adj_matrix_np.sum(axis=0)) / len(adj_matrix)
        print(adj_matrix)
        print('AVG # Parents:', avg_nr_parents)

        print()
        print('---- Parameter Learning ----')
        cptList = learn_parameters(adj_matrix, train_sample)
        p += 0.2
        print()

    print('Finished!')


def test_K(max_K, train_sample, scoreFunction):
    K = 1
    while K <= max_K:
        print('---- Structure Learning for K=', K, ' ----')
        score, adj_matrix = learn_structure(K, train_sample, scoreFunction)
        adj_matrix_np = np.array(adj_matrix)
        avg_nr_parents = sum(adj_matrix_np.sum(axis=0)) / len(adj_matrix)
        print(adj_matrix)
        print('AVG # Parents:', avg_nr_parents)
        K += 1
        print()

    print('Finished!')


def store_data(filename, itemlist):
    with open(filename, 'wb') as fp:
        pickle.dump(itemlist, fp)


def load_data(filename):
    with open(filename, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist


if __name__ == '__main__':
    all_data = read_data('wine.csv')
    print('Dataset size:', len(all_data))
    labels = all_data.pop(0)

    # TEST for p function
    # test_p(all_data, structurelearning.K2ScoreLogs)
    # Fix dataset and test for K
    # train_sample, test_sample = get_train_test_split(all_data, 0.6)
    # store_data('train_sample_70p', train_sample)
    # store_data('test_sample_70p', test_sample)

    train_sample = load_data('train_sample_70p')
    test_sample = load_data('test_sample_30p')
    print('Train and test dataset size: ', len(train_sample), ',', len(test_sample))
    # Expected input to the algorithm is that the arrays are the observations PER VARIABLE, therefore I need to
    # transpose the original data The reason for that is mainly because the algorithms were first developed from the
    # example shown in the lecture
    train_sample = list(map(list, zip(*train_sample)))
    # test_K(5, train_sample, structurelearning.BICScore)

    print('---- Structure Learning ----')
    # score, adj_matrix = learn_structure(2, train_sample, structurelearning.K2ScoreLogs)
    # adj_matrix_np = np.array(adj_matrix)
    # store_data('adj_matrix_wine_70p_K2Score', adj_matrix)
    # avg_nr_parents = sum(adj_matrix_np.sum(axis=0)) / len(adj_matrix)
    # print(adj_matrix)
    # print(score)
    # print('AVG # Parents:', avg_nr_parents)

    # score, adj_matrix = learn_structure(3, train_sample, structurelearning.K2ScoreLogs)
    # score, adj_matrix = learn_structure(4, train_sample, structurelearning.K2ScoreLogs)
    print()
    print('---- Parameter Learning ----')
    # adj_matrix = load_data('adj_matrix_wine_70p_K2Score')
    # cptList = learn_parameters(adj_matrix, train_sample)
    # store_data('cptList_wine_K2score', cptList)
    # print('CPT List:', len(cptList))

    #  print(cptList)
    print()
    print('---- Inference Accuracy Evaluation ----')
    adj_matrix = load_data('adj_matrix_wine_70p_K2Score')
    cptList = load_data('cptList_wine_K2score')
    model = (adj_matrix, cptList)
    print(cptList)

    # reduced_factors = variableelimination.variableElimination(11, [(1, 0)], model)
