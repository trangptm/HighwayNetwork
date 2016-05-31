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

The model was used in the highway network paper:
____________________________________________________________________________________________________
Layer (type)                       Output Shape        Param #     Connected to
====================================================================================================
reshape_1 (Reshape)                (None, 1, 28, 28)   0           reshape_input_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)                (None, 1, 28, 28)   0           reshape_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)    (None, 32, 24, 24)  832         dropout_1[0][0]
____________________________________________________________________________________________________
conv2dhighway_2 (Conv2DHighway)    (None, 32, 24, 24)  18496       convolution2d_1[0][0]
____________________________________________________________________________________________________
conv2dhighway_3 (Conv2DHighway)    (None, 32, 24, 24)  18496       conv2dhighway_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)      (None, 32, 12, 12)  0           conv2dhighway_3[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)                (None, 32, 12, 12)  0           maxpooling2d_1[0][0]
____________________________________________________________________________________________________
conv2dhighway_4 (Conv2DHighway)    (None, 32, 12, 12)  18496       dropout_2[0][0]
____________________________________________________________________________________________________
conv2dhighway_5 (Conv2DHighway)    (None, 32, 12, 12)  18496       conv2dhighway_4[0][0]
____________________________________________________________________________________________________
conv2dhighway_6 (Conv2DHighway)    (None, 32, 12, 12)  18496       conv2dhighway_5[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)      (None, 32, 6, 6)    0           conv2dhighway_6[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)                (None, 32, 6, 6)    0           maxpooling2d_2[0][0]
____________________________________________________________________________________________________
conv2dhighway_7 (Conv2DHighway)    (None, 32, 6, 6)    18496       dropout_3[0][0]
____________________________________________________________________________________________________
conv2dhighway_8 (Conv2DHighway)    (None, 32, 6, 6)    18496       conv2dhighway_7[0][0]
____________________________________________________________________________________________________
conv2dhighway_9 (Conv2DHighway)    (None, 32, 6, 6)    18496       conv2dhighway_8[0][0]
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)      (None, 32, 3, 3)    0           conv2dhighway_9[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)                (None, 32, 3, 3)    0           maxpooling2d_3[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)                (None, 288)         0           dropout_4[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                    (None, 10)          2890        flatten_1[0][0]
====================================================================================================
Total params: 151690
____________________________________________________________________________________________________
