# Supervised Hebbian Learning Algorithm

This repository contains the code that was used in the paper I wrote on the Supervised Hebbian Algorithm.
It contains the following: The main entry file, The base network class, the MNIST loader class, activation functions, MNIST data set, and the paper PDF file.

## Getting Started

### Prerequisites

You will need a recent version of Python. The project was written in Python 3.7 but you might be able to run it with a lower version.
You will need the cPickle library for reading the MNIST data set.

### Installing

Just download the zip file and extract the content into a Python directory of your choice. Run the main file and you should get results
for the default network settings.

## Experimenting with the model

You can tweak the hyper parameters of the network from the main file. Things like : network layers and sizes, learning rates, etc.
You can also inherit from the base class of the network and override the feedforward_with_learning method to implement your own.
Feel free to contact me if you need have any questions or suggestions.

## Built With

* [Miniconda](http://anaconda.com)

## Contributing

Please message me if you want to contribute to this project.

## Authors

* **Rafi Qumsieh** - *Initial work* - (https://github.com/rafiqumsieh0)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* The initial code was taken from Michael Nielsen's book "Neural networks and deep learning", so special thanks to him.

