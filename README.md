# HighwayNetwork

This project is my codes for Highway network using Keras with Theano backend.
More information about the model can be found in:

[Training very deep network](http://papers.nips.cc/paper/5850-training-very-deep-networks)

There are two versions: Highway networks for vector data and Convolutional Highway networks for spatial data.
Highway networks for vector data is already implemented in Keras layers.
Convolutional Highway layer is written based on Keras Highway and Convolutional layers.

I propose to modify the Highway networks so that the parameters can be shared among layers. This reduces the number of parameters.
To turn on the option parameter sharing, the scripts highway_training.py and convhighway_training.py can have an option "-shared x", where x = 1 means parameters are shared and x = 0 for the original network.

Experiments show that for mnist dataset, the original model outpeforms the model with parameter sharing while for some vector dataset, parameter sharing version works better (results of miniboo and sensorless datasets)

Experiment logs are in /log/ folder. For mnist dataset, I ran and reported the result after 20 epochs.
