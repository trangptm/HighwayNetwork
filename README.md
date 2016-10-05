# HighwayNetwork

This project is my codes for Highway network using Keras with Theano backend.
More information about the model can be found in:

[Training very deep network](http://papers.nips.cc/paper/5850-training-very-deep-networks)

A Highway network layer is a linear combination of the previous layer and the current activation.

h^t = g * h^t + (1-g) * h^(t-1)

where g is a sigmoid function of h^(t-1).

An interesting point is that the bias of g is initialized with a large negative number. This implies g -> 0. That means in early learning stage, the model prefers linear part, making the learning easier.

There are two versions: Highway networks for vector data and Convolutional Highway networks for spatial data.
Highway networks for vector data is already implemented in Keras layers.
Convolutional Highway layer is written based on Keras Highway and Convolutional layers. The convolutional highway model used in the paper is in file conv_highway_model.txt

I propose to modify the Highway networks so that the parameters can be shared among layers. This enables the model to learn high level of abstraction of the data while reduces the number of parameters. Parameter sharing in Highway networks may fit with small and medium datasets.

To turn on the option parameter sharing, the scripts highway_training.py and convhighway_training.py can have an option "-shared x", where x = 1 means parameters are shared and x = 0 for the original network.

Experiments show that for mnist dataset, the original model outpeforms the model with parameter sharing while for some vector dataset, parameter sharing version works better (results of miniboo and sensorless datasets).

Experiment logs are in /log/ folder. For mnist dataset, I ran and reported the result after 20 epochs.
