import numpy as np
import math
import activation_functions
import matplotlib.pyplot as plt


class BaseNetwork(object):
    """NETWORK PARAMS: SIZES, BIASES THRESHOLD, ETA_LTP, ETA_LTD, ETA_LTD2, ETA_LTD3, RESET_WEIGHT_POSITIVE, RESET_WEIGHT_NEGATIVE, TRAINING DATA, TEST DATA """
    def __init__(self, sizes, training_data, test_data, tanh_activations_coefficient, tanh_weights_coefficient
                 , weights, biases, eta_ltp, eta_ltd, eta_ltd2, create_new_connection_threshold,
                 weight_reset_value, connectivity_factors, start_learning_at_index
                 ):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = []
        self.initialize_biases(biases)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.connectivity_factors = connectivity_factors
        self.assert_sizes_match(biases, weights, connectivity_factors)
        self.initialize_weights(weights)
        self.training_data = training_data
        self.test_data = test_data
        self.results = [0]*10
        self.tanh_activations_coefficient = tanh_activations_coefficient
        self.tanh_weights_coefficient = tanh_weights_coefficient
        self.eta_ltp = eta_ltp
        self.eta_ltd = eta_ltd
        self.eta_ltd2 = eta_ltd2
        self.create_new_connection_threshold = create_new_connection_threshold
        self.weight_reset_value = weight_reset_value
        self.start_learning_at_index = start_learning_at_index

    def assert_sizes_match(self, biases, weights, connectivity_factors):
        size = len(self.sizes)
        assert len(biases) == size - 1, "Biases list does not match size of the network"
        assert len(weights) == size - 1, "Weights list does not match size of the network"
        assert len(connectivity_factors) == size - 1, "Connectivity factors list does not match size of the network"

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        a_list = [a]
        for layer_index, (b, w) in enumerate(zip(self.biases, self.weights)):
            activations = [0] * self.sizes[layer_index + 1]
            for m in range(0, self.sizes[layer_index + 1]):
                if layer_index == self.num_layers-1:
                    activations[m] = activation_functions.relu(np.dot(w[m], a_list[-1]) - b[m])
                else:
                    activations[m] = activation_functions.modified_tanh(np.dot(w[m], a_list[-1]) - b[m], self.tanh_activations_coefficient)
            a_list.append(activations)
        return a_list[-1]

    def feedforward_test(self, layer, in_put, label):
        """Return the output of the network if ``a`` is input and display the activations in the specified layer and the last output layer"""
        a_list = [in_put]
        index = 0
        for i in range(0, len(label)):
            if label[i] == 1:
                index = i
                break
        for layer_index, (b, w) in enumerate(zip(self.biases, self.weights)):
            activations = [0] * self.sizes[layer_index + 1]
            for m in range(0, self.sizes[layer_index + 1]):
                if layer_index == self.num_layers - 1:
                    activations[m] = activation_functions.relu(np.dot(w[m], a_list[-1]) - b[m])
                else:
                    activations[m] = activation_functions.modified_tanh(np.dot(w[m], a_list[-1]) - b[m], self.tanh_activations_coefficient)
            if layer_index == layer:
                pixels = np.array(a_list[layer])
                size = int(math.sqrt(self.sizes[layer_index]))
                pixels = pixels.reshape((size, size))
                plt.title('Label is {label}'.format(label=index))
                plt.imshow(pixels, cmap='gray')
                plt.show()
            a_list.append(activations)
        print("Predicted Output From Test is : {0}".format(a_list[-1]))
        return a_list[-1]

    def feedforward_with_learning(self, a, target_output, asynch=False):
        """Return the output of the network if ``a`` is input while modifying the weights as we go through each layer"""

        a_list = [a]
        for layer_index, (b, w) in enumerate(zip(self.biases, self.weights)):
            if layer_index + 1 == self.num_layers - 1:
                a_list.append(target_output)
            else:
                activations = [0] * self.sizes[layer_index + 1]
                for m in range(0, self.sizes[layer_index + 1]):
                    if layer_index == self.num_layers - 1:
                        activations[m] = activation_functions.relu(np.dot(w[m], a_list[-1]) - b[m])
                    else:
                        activations[m] = activation_functions.modified_tanh(np.dot(w[m], a_list[-1]) - b[m], self.tanh_activations_coefficient)
                a_list.append(activations)
            second_layer_size, first_layer_size = w.shape
            new_weights = w.copy()
            new_weight = 0.0
            for j in range(0, second_layer_size):
                if layer_index < self.start_learning_at_index:
                    continue
                for i in range(0, first_layer_size):
                    x = a_list[-2][i]
                    y = a_list[-1][j]
                    if x * y >= self.create_new_connection_threshold:
                        if w[j][i] != 0:
                            dw = self.eta_ltp * x * y
                            new_weight = activation_functions.relu_tanh(w[j][i] + dw, self.tanh_weights_coefficient)
                        else:
                            new_weight = 0.50
                    else:
                        dw = self.eta_ltd * x * y
                        new_weight = activation_functions.relu_tanh(w[j][i] - dw, self.tanh_weights_coefficient)
                        if new_weight < 0:
                            new_weight = 0
                    new_weights[j][i] = new_weight
            if asynch:
                self.weights[layer_index] = new_weights

    def evaluate(self, test_data):
        """Take the test data, and for each data point, run its input x through the network and compare to the actual output y.
        Count all correct classifications and return that count. Also count the correct classifications for each digit and store it in self.results"""
        test_results = []
        for (x, y) in test_data:
            predicted_y = self.feedforward(x)
            test_results.append((np.argmax(predicted_y), y))
        summation = 0
        for index, test_result in enumerate(test_results):
            x = test_result[0]
            y = test_result[1]
            summation += int(x == y)
            self.results[x] = self.results[x]+1
        return summation

    def train(self, num_epochs):
        """Train the network using training data with the specified number of epochs. """
        print("STARTED TRAINING")
        for i in range(0, num_epochs):
            print("EPOCH #{0}".format(i+1))
            for index, training_datum in enumerate(self.training_data):
                print("DATA POINT #{0}".format(index))
                self.feedforward_with_learning(training_datum[0], training_datum[1], True)

    def test_on_data(self, testing_data=None):
        """Test the network's performance using the supplied testing data using the evaluate function defined above."""
        length = 0
        if testing_data is None:
            testing_data = self.test_data
            length = 10000
        else:
            length = len(testing_data)
        self.results = [0] * 10
        evaluation = self.evaluate(testing_data)
        print("Correct {0} / {1}".format(evaluation, length))
        print("Accuracy : {0}".format(round(evaluation*100.0/length, 2)))
        return evaluation

    def initialize_weights(self, weights):
        """Initialize all weights based on the supplied weights list. It is a list of weight objects that contain information
        about the weight type {"POOLING","DENSE"} and the desired weights values for that layer. Neurons in the same layer are
        initialized to the same value."""
        for weight_matrix_index, weight_matrix in enumerate(self.weights):
            connection_type = weights[weight_matrix_index]["type"]
            weights_value = weights[weight_matrix_index]["value"]

            first_layer_size = self.sizes[weight_matrix_index]
            second_layer_size = self.sizes[weight_matrix_index+1]
            sqrt_first_layer = math.floor(pow(first_layer_size, 0.5))
            sqrt_second_layer = math.floor(pow(second_layer_size, 0.5))
            ratio = math.floor(sqrt_first_layer/sqrt_second_layer)

            for i in range(0, first_layer_size):
                row1 = math.floor(i/sqrt_first_layer)
                column1 = i % sqrt_first_layer
                for j in range(0, second_layer_size):
                    row2 = math.floor(j/sqrt_second_layer)
                    column2 = j % sqrt_second_layer

                    if connection_type == "POOLING":
                        if abs(math.floor(row1 / ratio) - row2) < 1 and abs(math.floor(column1 / ratio) - column2) < 1:
                            weight_matrix[j][i] = np.random.choice([weights_value], p=[1 / 1])
                        else:
                            weight_matrix[j][i] = 0
                    elif connection_type == "DENSE":
                        weight_matrix[j][i] = np.random.choice([weights_value], p=[1 / 1])

    def initialize_biases(self, biases):
        """Initialize all biases using the supplied biases list. It is a list that contains biases values for each layer.
        Neurons in the same layer have same bias values."""
        for layer_size, bias in zip(self.sizes[1:], biases):
            self.biases.append(np.full((layer_size, 1), bias))

    def prune_small_weights_to(self, threshold, portion):
        """Initialize all biases using the supplied biases list. It is a list that contains biases values for each layer.
        Neurons in the same layer have same bias values."""
        for w in self.weights[1:]:
            second_layer_size, first_layer_size = w.shape
            for j in range(0, second_layer_size):
                for i in range(0, first_layer_size):
                    if abs(w[j][i]) < threshold:
                        w[j][i] = portion * w[j][i]

    def reset_weight(self, weight):
        """Reset the weight to the specified self.weight_reset_value if the weight's absolute value is above 1."""
        if weight >= 1.0:
            return self.weight_reset_value
        elif weight <= -1.0:
            return -1*self.weight_reset_value
        return weight




