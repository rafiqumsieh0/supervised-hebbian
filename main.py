"""" Import mnist_loader which handles loading, formatting, choosing mnist data.
 base_network is the base class for our neural network and can be overridden by rewriting the function : feedforward_with_learning"""
import mnist_loader
import base_network

"""" INITIALIZE THE NETWORK'S PARAMETERS """

"""Define a sizes list of layers """
sizes = [784, 196, 10]
"""output activation coefficient is the coefficient of the input of the tanh function.
 It controls the curve of the tanh activation function """
output_activation_coefficient = 0.50
"""weight activation coefficient is the coefficient of the input of the tanh function for weight change. It controls the curve of the tanh activation function
and only applies if we use tanh for weight change. """
weight_activation_coefficient = 0.50
"""The value to reset the weights to if the weights cross 1 or -1. """
weight_reset_value = 0.90
"""The value to hard set the weight to if the weight is initially zero but the neurons are firing together strongly.
This will create a new connection between the neurons with the specified value. """
create_new_connection_threshold = 0.25
"""Weights is a list of dictionaries, each dictionary has two keys: type (POOLING, DENSE) and value which represents the value of the weight """
weights = [{"type": "POOLING", "value": 0.50}, {"type": "DENSE", "value": 0.00}]
"""Biases is list of values representing the biases in each layer"""
biases = [0.95, 0.00]
"""eta_ltp specifies the learning rate for the long-term potentiation process which happens when the neurons are firing together."""
eta_ltp = 0.01
"""eta_ltd specifies the learning rate for the long-term depression process which happens when the input fires but not the output"""
eta_ltd = 0.0005
"""eta_ltd2 specifies the learning rate for the long-term depression process which happens when the output fires but not the input"""
eta_ltd2 = 0.00
"""connectivity_factors determine how much the neurons are connected in a pooling layer. the larger the connectivity factor,
the more connections a neuron has to the previous layer."""
connectivity_factors = [1, 1]
"""Specifies the index of weights to start learning from"""
start_learning_at_index = 1
"""" INITIALIZE THE SCRIPT'S PARAMETERS """
num_images_per_digit = 60
accuracies = []
total = 0
"""" INITIALIZE THE NETWORK, TRAIN ON THE TRAINING DATA SET, THEN TEST ON THE SPECIFIED DATA SET """
num_images_per_digit_array = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
for j in range(0, len(num_images_per_digit_array)):
    num_images_per_digit_array[j] = num_images_per_digit*num_images_per_digit_array[j]
training_data, test_data, validation_data, raw_training_data = mnist_loader.pick_training_data_with(num_images_per_digit_array)
neural_network = base_network.BaseNetwork(sizes=sizes,
                                          training_data=training_data,
                                          test_data=test_data,
                                          tanh_activations_coefficient=output_activation_coefficient,
                                          tanh_weights_coefficient=weight_activation_coefficient,
                                          weights=weights,
                                          biases=biases,
                                          eta_ltp=eta_ltp,
                                          eta_ltd=eta_ltd,
                                          eta_ltd2=eta_ltd2,
                                          create_new_connection_threshold=create_new_connection_threshold,
                                          weight_reset_value=weight_reset_value,
                                          connectivity_factors=connectivity_factors,
                                          start_learning_at_index=start_learning_at_index)

"""Train with the specified number of epochs."""
neural_network.train(num_epochs=1)
""" Uncomment this to see the result for a specific layer on a specific data point"""
#neural_network.feedforward_test(layer=2, input=training_data[5][0], label=training_data[5][1])
""" Test on the provided testing data"""
evaluation = neural_network.test_on_data(testing_data=None)
""" Add testing results to the total accuracies list"""
accuracies.append(round(evaluation/100, 2))
""" Print the results for each digit"""
print(neural_network.results)
for accuracy in accuracies:
    total = total + accuracy
print(total/len(accuracies))