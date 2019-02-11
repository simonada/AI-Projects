Readme for VE593: AI Project III

Author: Simona Emilova Doneva

As requested the submitted files consist of the following:

1. Networks.py
- three flags are set at the beginning of the class:
	- verbose: enables additional outputs for debugging
	- monitor_test: will gather statistics on the test data
	- l1_regularization: will update the weights using L1 regularization
	
- contains the implementation of the required functions to enable network learning and extensions, including:
	- the constructor has two new parameters: activationFcns to enable different activation functions to be attached to a layer,
	and test_data to enable to gather performance statistics on this data set during training
	- training function: returns validation_accuracy, training_accuracy, testing_accuracy over the epochs
	- splitTrainData: returns batches of the specified size
	- updateWeights: updates the weights and biases of the Network
	- backprop: returns (gradients_weights, gradients_biases) for a single sample
	- evaluate: test the achieved accuracy of the network on the provided data set
	- dSquaredLoss: derivation function of the squared loss
	- activation functions and their derivatives for: sigmoid, tanh, relu and leaky relu

2. experiments.py
- the main function prepares the data sets and executes the experiments reported in the report
- the experiments functions include:
	- test_different_layer_architectures: initial selection of a baseline model
	- learning_rate_search: iterates over different learning rates
    - batch_size_search: iterates over different batch sizes
    - test_regularization_score: iterates over different regularization scores
    - activation_functions_search: iterates over different activation functions applied to the hidden layer
    - advanced_test_different_layer_depth_architectures: will test different depths for the network, default width of a layer is 15
    - advanced_test_different_layer_width_architectures: will test different widths for the network, one can specify the starting width,
    the maximum width, and by how much should the width increase at each step
    - test_baseline_fine_tuned_networks: final performance comparison of fine tuned vs. baseline model
- all of the functions will output performance statistics on train, test and validation data at the end