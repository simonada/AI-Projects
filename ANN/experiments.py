import Networks
import database_loader
import random
import time
from Networks import sigmoid
from Networks import tanh
from Networks import relu
from Networks import leaky_relu
import numpy as np


def print_time_output(start, end):
    hours, total_seconds = divmod(end - start, 3600)
    minutes, seconds = divmod(total_seconds, 60)
    print()
    print("Training finished, it took {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    return minutes, seconds, total_seconds


def activation_functions_search(baseline_network, train_data_tuples, valid_data_tuples, test_data_tuples):

    allowed_functions = [Networks.sigmoid, Networks.tanh, Networks.relu, Networks.leaky_relu]
    fct_names = []

    val_accuracies = []
    train_accuracies = []
    test_accuracies = []

    original_weights = baseline_network.weights
    original_biases = baseline_network.biases

    for act_fct in allowed_functions:
        print('Activation function', act_fct.__name__)
        fct_names.append(act_fct.__name__)

        network = Networks.Network([784, 20, 10], [None, act_fct, sigmoid], test_data_tuples)

        network.weights = original_weights
        network.biases = original_biases

        validation_accuracy, training_accuracy, test_accuracy = network.training(train_data_tuples, 30, 50, 10.0,
                                                                                 valid_data_tuples, 20)

        val_accuracies.append(validation_accuracy)
        train_accuracies.append(training_accuracy)
        test_accuracies.append(test_accuracy)
        print()

    print()
    print('Learning rate search final results')
    print('Activation functions ', fct_names)
    print('Train acc', train_accuracies)
    print('Train avg', [np.mean(item) for item in train_accuracies])
    print('Validation acc:', val_accuracies)
    print('Validation avg', [np.mean(item) for item in val_accuracies])
    print('Test acc', test_accuracies)
    print('Test avg', [np.mean(item) for item in test_accuracies])


def learning_rate_search(network, train_data_tuples, valid_data_tuples):
    # fixed:
    # epochs = 100
    # batch size = 1 , i.e. no batching
    # no regularization
    learning_rate = 0.01
    learning_rates = []
    val_acc_stats = []
    original_weights = network.weights
    original_biases = network.biases

    while learning_rate <= 15:
        print('Train with learning rate', learning_rate)
        # make sure it's influence by the previous tests
        network.weights = original_weights
        network.biases = original_biases

        validation_accuracy, training_accuracy, testing_acc = network.training(train_data_tuples, 10, 50, learning_rate,
                                                                               valid_data_tuples, 20)
        val_acc_stats.append(validation_accuracy)
        learning_rates.append(learning_rate)
        if learning_rate > 1:
            learning_rate *= 2
        else:
            learning_rate *= 5
        print()
    print()
    print('Learning rate search final results')
    print('Learning rates:', learning_rates)
    print('Validation acc:', val_acc_stats)

    return learning_rates, val_acc_stats


def batch_size_search(network, train_data_tuples, valid_data_tuples):
    # idea: plot validation accuracy vs time and choose the size that gives the most rapid improvement in performance
    # maximize overall speed
    batch_sizes = []
    val_acc_stats = []
    total_seconds = []
    learning_rates = [0.25]
    original_weights = network.weights
    original_biases = network.biases
    original_learning_rate = 0.25
    learning_rate = original_learning_rate

    for b_size in range(0, 100, 10):
        if b_size == 0:
            b_size = 1
        if b_size > 1:
            learning_rate = original_learning_rate * b_size  # Linear Scaling of the Learning Rate
        print('Learning rate', learning_rate)
        network.weights = original_weights
        network.biases = original_biases

        batch_sizes.append(b_size)
        start = time.time()
        validation_accuracy, training_accuracy = network.training(train_data_tuples, 5, b_size, learning_rate,
                                                                  valid_data_tuples, 0)
        end = time.time()
        mins, secs, total_sec = print_time_output(start, end)
        val_acc_stats.append(validation_accuracy)
        total_seconds.append(total_sec)
        learning_rates.append(learning_rate)

    print('Batch size performance statistics')
    print('Batch sizes:', batch_sizes)
    print('Learning rates', learning_rates)
    print('Validation acc:', val_acc_stats)
    print('Validation avg', [np.mean(item) for item in val_acc_stats])
    print('Seconds training took', total_seconds)


def test_regularization_score(network, train_data_tuples, valid_data_tuples):
    # validation_accuracy_no_reg, training_accuracy_no_reg, test_accuracy = network.training(train_data_tuples, 50, 50, 10.0, valid_data_tuples, 0)
    # print('Results without regularizatoin')
    # print('valid', validation_accuracy_no_reg)
    # print('train', training_accuracy_no_reg)
    # print('test', test_accuracy)

    val_accuracies = []
    train_accuracies = []
    test_accuracies = []
    lmda_scores = []
    original_weights = network.weights
    original_biases = network.biases

    for lmbda in range(0, 40, 10):
        if lmbda == 0:
            lmbda = 1
        network.weights = original_weights
        network.biases = original_biases
        validation_accuracy, training_accuracy, test_acc = network.training(train_data_tuples, 30, 50, 10.0,
                                                                            valid_data_tuples, lmbda)

        val_accuracies.append(validation_accuracy)
        train_accuracies.append(training_accuracy)
        lmda_scores.append(lmbda)
        test_accuracies.append(test_acc)

    print('Batch size performance statistics')
    print('Lambda scores:', lmda_scores)
    print('Train acc', train_accuracies)
    print('Train avg', [np.mean(item) for item in train_accuracies])
    print('Validation acc:', val_accuracies)
    print('Validation avg', [np.mean(item) for item in val_accuracies])
    print('Test acc', test_accuracies)
    print('Test avg', [np.mean(item) for item in test_accuracies])


def advanced_test_different_layer_width_architectures(train_data_tuples, valid_data_tuples, test_data_tuples):
    val_statistics = []
    train_statistics = []
    test_statistics = []
    total_seconds = []
    hl_size = []

    for hl in range(20, 100, 10):
        if hl == 0:
            continue
        print(hl)
        network = Networks.Network([784, hl, 10], [None, sigmoid, sigmoid], test_data_tuples)
        start = time.time()

        validation_accuracy, training_accuracy, test_accuracy = network.training(train_data_tuples, 10, 50, 10.0,
                                                                                 valid_data_tuples, 20)
        end = time.time()
        mins, secs, total_sec = print_time_output(start, end)

        total_seconds.append(total_sec)
        val_statistics.append(np.mean(validation_accuracy))
        train_statistics.append(np.mean(training_accuracy))
        test_statistics.append(np.mean(test_accuracy))
        hl_size.append(hl)

    print()
    print('HL Size')
    print(hl_size)
    print('Validation')
    print(val_statistics)
    print('Training')
    print(train_statistics)
    print('Testing')
    print(test_statistics)
    print('Final test data performance')
    print(network.evaluate(test_data_tuples))
    print('Time')
    print(total_seconds)


def advanced_test_different_layer_depth_architectures(train_data_tuples, valid_data_tuples, test_data_tuples):
    layers = [1, 2, 3, 4, 5]
    hidden_units = [15]  # , 50, 150, 500, 1000]

    val_statistics = np.zeros(shape=(len(layers), len(hidden_units)))
    train_statistics = np.zeros(shape=(len(layers), len(hidden_units)))
    test_statistics = np.zeros(shape=(len(layers), len(hidden_units)))
    total_seconds = []

    for hidden_layer_size in layers:
        for nr_hidden_units in hidden_units:
            network_structure = np.empty(shape=(hidden_layer_size + 2), dtype=int)
            activations = np.empty(shape=(hidden_layer_size + 2), dtype=object)
            # print(activations)
            activations[0] = None
            activations[-1] = Networks.sigmoid
            network_structure[0] = 784
            network_structure[-1] = 10

            for layer in range(1, len(network_structure) - 1):
                network_structure[layer] = nr_hidden_units
                activations[layer] = Networks.sigmoid

            final_network_architecture = list(network_structure)

            network = Networks.Network(final_network_architecture, activations, test_data_tuples)

            start = time.time()
            validation_accuracy, training_accuracy, test_accuracy = network.training(train_data_tuples, 40, 50, 10.0,
                                                                                     valid_data_tuples, 20)
            end = time.time()
            mins, secs, total_sec = print_time_output(start, end)
            total_seconds.append(total_sec)

            hl_size_case = layers.index(hidden_layer_size)
            nr_units_case = hidden_units.index(nr_hidden_units)
            val_statistics[hl_size_case][nr_units_case] = np.mean(validation_accuracy)
            train_statistics[hl_size_case][nr_units_case] = np.mean(training_accuracy)
            test_statistics[hl_size_case][nr_units_case] = np.mean(test_accuracy)
    print()
    print('Validation')
    print(val_statistics)
    print('Training')
    print(train_statistics)
    print('Testing')
    print(test_statistics)
    print('Final test data performance')
    print(network.evaluate(test_data_tuples))
    print('Time')
    print(total_seconds)


def test_different_layer_architectures(train_data_tuples, valid_data_tuples, test_data_tuples):
    network_no_hl = Networks.Network([784, 10], [None, sigmoid], test_data_tuples)
    hl_nodes = [0]
    val_accuracies = []
    train_accuracies = []
    test_accuracies = []

    validation_accuracy, training_accuracy, test_accuracy = network_no_hl.training(train_data_tuples, 5, 1, 1.0,
                                                                                   valid_data_tuples, 0)
    val_accuracies.append(np.mean(validation_accuracy))
    test_accuracies.append(np.mean(test_accuracy))
    train_accuracies.append(np.mean(training_accuracy))

    for hl in range(0, 25, 5):
        if hl == 0:
            continue
        network = Networks.Network([784, hl, 10], [None, sigmoid, sigmoid], test_data_tuples)
        validation_accuracy, training_accuracy, test_acc = network.training(train_data_tuples, 5, 1, 1.0,
                                                                            valid_data_tuples, 0)
        val_accuracies.append(np.mean(validation_accuracy))
        train_accuracies.append(np.mean(training_accuracy))
        test_accuracies.append(np.mean(test_acc))
        hl_nodes.append(hl)

    print('HL nodes', hl_nodes)
    print('Val Accuracy performance', val_accuracies)
    print('Train Accuracy performance', train_accuracies)
    print('Test Accuracy performance', test_accuracies)


def test_baseline_fine_tuned_networks(train_data_tuples, valid_data_tuples, test_data_tuples):
    baseline_network = Networks.Network([784, 20, 10], [None, sigmoid, sigmoid], test_data_tuples)
    original_weights = baseline_network.weights
    original_biases = baseline_network.biases

    start = time.time()
    validation_accuracy, training_accuracy, test_acc = baseline_network.training(train_data_tuples, 50, 1, 1.0,
                                                                                 valid_data_tuples, 0)
    end = time.time()
    mins, secs, total_sec = print_time_output(start, end)
    print('Validation')
    print(np.mean(validation_accuracy))
    print('Training')
    print(np.mean(training_accuracy))
    print('Testing')
    print(np.mean(test_acc))
    print('Final test data performance')
    print(baseline_network.evaluate(test_data_tuples))
    print('Time to complete in seconds:', total_sec)

    print()
    print('Fine-tuned network performance:')
    baseline_network.weights = original_weights
    baseline_network.biases = original_biases
    start = time.time()
    validation_accuracy, training_accuracy, test_acc = baseline_network.training(train_data_tuples, 50, 50, 10.0,
                                                                                 valid_data_tuples, 20)
    end = time.time()
    mins, secs, total_sec = print_time_output(start, end)
    print('Validation')
    print(np.mean(validation_accuracy))
    print('Training')
    print(np.mean(training_accuracy))
    print('Testing')
    print(np.mean(test_acc))
    print('Final test data performance')
    print(baseline_network.evaluate(test_data_tuples))
    print('Time to complete in seconds:', total_sec)



def main():
    training_data, validation_data, test_data = database_loader.load_data()

    # -- Data Preparation ---
    train_data = list(zip(*training_data))
    x, y = train_data
    # list of (input, target) training samples
    train_data_tuples = list(zip(x, y))

    test_data = list(zip(*test_data))
    x_test, y_test = test_data
    # list of (input, target) training samples
    test_data_tuples = list(zip(x_test, y_test))

    valid_data = list(zip(*validation_data))
    x_val, y_val = valid_data
    # list of (input, target) training samples
    valid_data_tuples = list(zip(x_val, y_val))

    # --- EXPERIMENTS ---
    test_different_layer_architectures(train_data_tuples, valid_data_tuples, test_data_tuples)
    baseline_network = Networks.Network([784, 20, 10], [None, sigmoid, sigmoid], test_data_tuples)

    learning_rate_search(baseline_network, train_data_tuples, valid_data_tuples)
    batch_size_search(baseline_network, train_data_tuples, valid_data_tuples)
    test_regularization_score(baseline_network, train_data_tuples, valid_data_tuples)

    activation_functions_search(baseline_network, train_data_tuples, valid_data_tuples, test_data_tuples)
    advanced_test_different_layer_depth_architectures(train_data_tuples, valid_data_tuples, test_data_tuples)
    advanced_test_different_layer_width_architectures(train_data_tuples, valid_data_tuples, test_data_tuples)

    test_baseline_fine_tuned_networks(train_data_tuples, valid_data_tuples, test_data_tuples)
    # test higher width and depth in network
    deeper_network = Networks.Network([784, 50, 50, 10], [None, sigmoid, relu, sigmoid], test_data_tuples)
    validation_accuracy, training_accuracy, test_acc = deeper_network.training(train_data_tuples, 30, 40, 5.0, valid_data_tuples, 30)
    print('Validation', np.mean(validation_accuracy), validation_accuracy)
    print('Training', np.mean(training_accuracy), training_accuracy)
    print('Testing', np.mean(np.mean(test_acc)), test_acc)


if __name__ == '__main__':
    main()
