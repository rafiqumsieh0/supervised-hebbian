"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

# Libraries
# Standard library
import _pickle as cpickle
import gzip
import math

# Third-party libraries
import numpy as np


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cpickle.load(f, encoding='iso-8859-1')
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data


def pick_training_data_with(num_images_per_digit_array):
    """Specify an array/list of the number of images for each digit for training, and this method will
    return a tuple of (reduced data, test data, validation data, raw reduced data)
    where reduced data is the data matching your query array where each entry is formatted and the output
    in each entry is a one-hot vector for the specified digit.
    raw reduced data is the same as reduced data but the output is simply the digit's value instead of a vector. We use
    this for testing. Note that the data is not randomized. You can get them randomized by simply shuffling the data
    before picking.
    """
    training_data, validation_data, test_data = load_data_wrapper()
    reduced_data = []
    raw_reduced_data = []
    results_counter = [0] * 10
    total = sum(num_images_per_digit_array)
    for training_datum in training_data:
        y = training_datum[1]
        index = 0
        for i in range(0, len(y)):
            if y[i] == 1:
                index = i
                break
        if results_counter[index] < num_images_per_digit_array[index]:
            reduced_data.append(training_datum)
            raw_reduced_data.append((training_datum[0], index))
            results_counter[index] = results_counter[index] + 1
        if len(reduced_data) == total:
            break
    return reduced_data, test_data, validation_data, raw_reduced_data


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
