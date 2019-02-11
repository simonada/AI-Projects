import pickle
import gzip
import numpy as np


def load_data():
    """
    :returns:
        training_data:  a list containing 50,000 2-tuples (x, y).
            x is a 784-dimensional numpy.ndarray containing the input image (since the size of each image is 28*28d).
            y is a 10-dimensional numpy.ndarray representing the unit vector corresponding to the correct digit for x.
        validation_data： list containing 10,000 2-tuples (x, y).
        test_data： list containing 10,000 2-tuples (x, y).
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    tr_d, va_d, te_d = pickle.load(f, encoding="latin1")
    f.close()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_results = [vectorized_result(y) for y in va_d[1]]
    validation_data = zip(validation_inputs, validation_results)

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_results)
    return training_data, validation_data, test_data


def vectorized_result(j):
    """
    Used to convert a digit (0...9) into a corresponding desired output from the neural network.
    :return: a 10-dimensional unit vector with a 1.0 in the jth position and zeroes elsewhere.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
