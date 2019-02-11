import numpy as np
import random

verbose = False
monitor_test = True
l1_regularization = False


class Network(object):

    def __init__(self, sizes, activationFcns, test_data):
        """
        :param: sizes: a list containing the number of neurons in the respective layers of the network.
                See project description.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.activation_functions = activationFcns

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.test_data = test_data

    def inference(self, x):
        """
        :param: x: input of ANN
        :return: the output of ANN with input x, a 1-D array
        """
        # print('Original input', x)
        # print('Number of layers', len(self.weights))
        L = len(self.weights)
        W = self.weights
        B = self.biases
        activation = x
        for layer in range(1, L + 1):
            # print('Layer Input', input_z.shape)
            # flatten the dimensions of the biases, otherwise the sum is not pointwise
            hidden_layer = (np.dot(W[layer - 1], activation)) + B[layer - 1]  # .ravel()
            activation = self.activation_functions[layer](hidden_layer)

            if verbose:
                print('Weights', W[layer - 1], 'biases', B[layer - 1])
                print('Input after weighting', hidden_layer.shape, hidden_layer)
                print('Input after activation', activation)

        # print('Final output', input_z)
        return activation

    def training(self, trainData, T, n, alpha, validationData=None, lmbda = 0):
        """
        trains the ANN with training dataset using stochastic gradient descent
        :param trainData: a list of tuples (x, y) representing the training inputs and the desired outputs.
        :param T: total number of iteration
        :param n: size of mini-batches
        :param alpha: learning rate
        """
        self.N = len(trainData) # needed for Regularization
        epochs = []
        training_accuracy = []
        validation_accuracy = []
        max_validation_score = 0
        count_epochs_without_improvement = 0
        testing_accuracy = []

        for iterations in range(T):
            print('Epoch', iterations)
            epochs.append(iterations)
            batches = self.splitTrainData(trainData, n)
            print('Batches', len(batches))
            # Epoch = iteration over the whole data set
            for batch in batches:
                self.updateWeights(batch, alpha, lmbda)
            if validationData:
                val_accuracy = self.evaluate(validationData)
                print("Performance on validation data: {0} / {1} ({2})".format(val_accuracy, len(validationData),
                                                                               val_accuracy / len(validationData)))
                validation_accuracy.append((val_accuracy / len(validationData) * 100))
                if val_accuracy > max_validation_score:
                    max_validation_score = val_accuracy
                    count_epochs_without_improvement = 0
                else:
                    count_epochs_without_improvement += 1
                    if count_epochs_without_improvement == 10:
                        print('Early stopping after ten epochs no improvement in accuracy.')
                        print('Max accuracy', max_validation_score)
                        print('Epochs', epochs)
                        return validation_accuracy, training_accuracy, testing_accuracy

            training_accuracy.append((self.evaluate(trainData) / len(trainData)) * 100)
            if monitor_test:
                testing_accuracy.append((self.evaluate(self.test_data) / len(self.test_data)) * 100)

        # print()
        # print('Performance statistics')
        # print('Epochs', epochs)
        # print('Training acc')
        # print(training_accuracy)
        # print('Validation acc')
        # print(validation_accuracy)

        return validation_accuracy, training_accuracy, testing_accuracy

    def splitTrainData(self, trainData, n):
        random.shuffle(trainData)
        batches = [trainData[k:k + n] for k in range(0, len(trainData), n)]
        return batches

    def updateWeights(self, batch, alpha, lmbda = 0):
        """
        called by 'training', update the weights and biases of the ANN
        :param batch: mini-batch, a list of pair (x, y)
        :param alpha: learning rate
        """

        biases_aggregated_updates = []
        weights_aggregated_updates = []
        batch_size = len(batch)
        # 1. Prepare to aggregate the gradients computed for each sample in the mini-batch
        # aggregated into an list of array of the same dimensions as the original weights and biases
        # initialized with zeros

        for w in self.weights:
            weights_aggregated_updates.append(np.zeros(w.shape))
        for b in self.biases:
            biases_aggregated_updates.append(np.zeros(b.shape))

        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            # zip them together to be able to iterate over them pair wise and update the W and B layer by layer
            weights_pairs = zip(weights_aggregated_updates, delta_nabla_w)
            biases_pairs = zip(biases_aggregated_updates, delta_nabla_b)

            biases_aggregated_updates = [batch_bias + sample_bias_delta for batch_bias, sample_bias_delta in
                                         biases_pairs]
            weights_aggregated_updates = [batch_weigths + sample_weights_delta for batch_weigths, sample_weights_delta
                                          in weights_pairs]
        # Final Batch update
        weights_final_pairs = zip(self.weights, weights_aggregated_updates)
        biases_final_pairs = zip(self.biases, biases_aggregated_updates)

        updated_weights = []
        updated_biases = []
        # Average the updates by dividing the learning rate by the number of samples in the mini batch
        alpha = (alpha / batch_size)
        for old_weights, batch_gradients_weights in weights_final_pairs:
            if l1_regularization:
                #print('Performing L1 regularization:')
                regularized_weights = old_weights - alpha * (lmbda/self.N)*np.sign(old_weights)   # if reg term is zero, the results are the same as if no reg. is applied
            else:
                regularized_weights = old_weights * (1 - alpha * (lmbda/self.N))  # if reg term is zero, the results are the same as if no reg. is applied
            updated_weights.append(regularized_weights - alpha * batch_gradients_weights)

        for old_biases, batch_gradients_biases in biases_final_pairs:
            updated_biases.append(old_biases - alpha * batch_gradients_biases)
        self.weights = updated_weights
        self.biases = updated_biases

    def backprop(self, x, y):
        """
        called by 'updateWeights'
        :param: (x, y): a tuple of batch in 'updateWeights'
        :return: a tuple (nablaW, nablaB) representing the gradient of the empirical risk for an instance x, y
                nablaW and nablaB follow the same structure as self.weights and self.biases
        """
        W = self.weights
        nr_layers = len(W)
        # Data structure for the final outputs
        gradients_weights = []
        gradients_biases = []
        for w in self.weights:
            gradients_weights.append(np.zeros(w.shape))
        for b in self.biases:
            gradients_biases.append(np.zeros(b.shape))

        layers_activations, layers_outputs = self.forward_pass(x)
        # First step get loss derivatives of the final layer
        # Matrix to store the deltas for each layer, should have a shape dependent on
        # the number of HL and the number of nodes in each HL
        prediction = layers_activations[-1]
        last_layer_outputs = layers_outputs[-1]

        if self.activation_functions[-1].__name__ == "sigmoid":
            last_layer_output_derivatives = sigmoid_prime(last_layer_outputs)
        elif self.activation_functions[-1].__name__ == "tanh":
            last_layer_output_derivatives = tanh_prime(last_layer_outputs)
        elif self.activation_functions[-1].__name__ == "relu":
            last_layer_output_derivatives = relu_prime(last_layer_outputs)
        elif self.activation_functions[-1].__name__ == "leaky_relu":
            last_layer_output_derivatives = leaky_relu_prime(last_layer_outputs)

        last_layer_delta = dSquaredLoss(prediction, y) * last_layer_output_derivatives

        if verbose:
            print('Squared loss derivatives', dSquaredLoss(prediction, y))
            print('Z last layer derivatives', last_layer_output_derivatives)
            print('Deltas last layer', last_layer_delta)

        # Next compute the derivatives w.r.t. to the errors at each layer, i.e. by how much does each node contribute
        # logic for delta = weights of layer above * delta of layer above dot sigmoid derivative of node output?

        layers_deltas = np.zeros(nr_layers, dtype=object)
        layers_deltas[-1] = last_layer_delta

        # Compute the deltas for each layer based on the deltas of the layer above it (iterate backwards)
        for layer in range(nr_layers - 2, -1, -1):
            w_previous_layer = W[layer + 1]
            deltas_previous_layer = layers_deltas[layer + 1]
            error_contributions_per_node = np.dot(w_previous_layer.T, deltas_previous_layer)

            if self.activation_functions[layer + 1].__name__ == "sigmoid":
                slope_derivatives_per_node = sigmoid_prime(layers_outputs[layer])
            elif self.activation_functions[layer + 1].__name__ == "tanh":
                slope_derivatives_per_node = tanh_prime(layers_outputs[layer])
            elif self.activation_functions[layer + 1].__name__ == "relu":
                slope_derivatives_per_node = relu_prime(layers_outputs[layer])
            elif self.activation_functions[layer + 1].__name__ == "leaky_relu":
                slope_derivatives_per_node = leaky_relu_prime(layers_outputs[layer])

            layer_delta = error_contributions_per_node * slope_derivatives_per_node
            layers_deltas[layer] = layer_delta

        # Final step, computing the deltas for weights updates at each layer
        for i in range(nr_layers):
            # weights
            activation = layers_activations[i]
            delta = layers_deltas[i]
            gradient_w = np.dot(delta, activation.T)
            gradients_weights[i] = gradient_w

        gradients_biases = layers_deltas

        return (gradients_weights, gradients_biases)

    def forward_pass(self, x):
        W = self.weights
        B = self.biases
        L = len(W)
        # print('---Forward pass---')
        activation = x
        activations_array = [x]  # the first activation is the input itself
        function_outputs_array = []
        for layer in range(1, L + 1):
            hidden_layer_output = (np.dot(W[layer - 1], activation)) + B[layer - 1]  # .ravel()
            function_outputs_array.append(hidden_layer_output)
            activation = self.activation_functions[layer](hidden_layer_output)
            activations_array.append(activation)
        return activations_array, function_outputs_array

    def evaluate(self, data):
        """
        :param data: dataset, a list of tuples (x, y) representing the training inputs and the desired outputs.
        :return: the number of correct predictions of the current ANN on the input dataset.
                The prediction of the ANN is taken as the argmax of its output
        """
        accuracy = 0
        count = 0
        for x, y in data:
            count += 1
            probabilities = self.inference(x)
            max_probability_class_id = probabilities.argmax(axis=0)
            # check if prediction matches the target
            if y[max_probability_class_id] == 1:
                accuracy += 1
        return accuracy


# activation functions together with their derivative functions:
def dSquaredLoss(a, y):
    """
    :param a: vector of activations output from the network
    :param y: the corresponding correct label
    :return: the vector of partial derivatives of the squared loss with respect to the output activations
    """
    # assuming that we measure the L(a, y) by sum(1/2*square(y_i - a_i)) for all i parameters, so that the two cancels out
    # for each partial derivation L/a_i we have  1/2*square(a_j -y_j) = 0 where j != i
    # partial derivation for a single 1/2*square(a_i -y_i) = 1/2 * 2 * (y_i - a_i) * -1 = a_i - y_i
    # (a_i - y_i) gives the contribution of the final output per node to the total error

    return (a - y)  # * a_derivatives


def squaredLoss(x, y):
    print('prediction', x)
    print('target', y)
    return np.sum(np.square(x - y))


def sigmoid(z):
    """The sigmoid function"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    return (1.0 - np.exp(-2 * z)) / (1.0 + np.exp(-2 * z))


def tanh_prime(z):
    return 1 - np.square(tanh(z))


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    return (z > 0) * 1  # if the element is greater than zero, it'll be set to one, otherwise zero


def leaky_relu(z):
    # if value above 0, keep the value, else replace value with beta * value
    beta = 0.01
    return np.where(z > 0, z, - beta * z)


def leaky_relu_prime(z):
    beta = 0.01
    dz = np.ones_like(z)
    dz[z < 0] = - beta
    return dz


def main():
    # ref. for the example: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    print('Running Networks example')
    net = Network([2, 2, 2])
    sample_weigths_l1 = np.array([[0.15, 0.25],
                                  [0.2, 0.30]])
    sample_weigths_l2 = np.array([[0.40, 0.50],
                                  [0.45, 0.55]])
    net.weights[0] = sample_weigths_l1
    net.weights[1] = sample_weigths_l2
    net.biases[0] = np.array([[0.35], [0.35]])
    net.biases[1] = np.array([[0.60], [0.60]])

    x = np.array([0.05, 0.10])  # input
    y = np.array([0.01, 0.99])  # target

    nablaW, nablaB = net.backprop(x, y)
    print('Weights')
    print(nablaW)
    print('Biases')
    print(nablaB)

    # for i in range(10):
    #   nablaW, nablaB = net.backprop(x, y)
    #   print(nablaW)
    #   net.weights -= 0.5 * nablaW
    #   net.biases -= 0.5 * nablaB


if __name__ == '__main__':
    main()
