from keras.models import Sequential
from ConvHighway import *
from keras.constraints import *

def create_highway(n_layers=10, dim=50, input_shape=None, n_classes=2, shared=1):
    act = 'relu'
    top_act = 'sigmoid' if n_classes == 1 else 'softmax'

    trans_bias = - n_layers // 10
    shared_highway = Highway(activation=act, init='orthogonal', W_constraint=maxnorm(2), transform_bias=trans_bias)

    def _highway():
        if shared == 1: return shared_highway
        return Highway(activation=act, init='orthogonal', W_constraint=maxnorm(2), transform_bias=trans_bias)

    model = Sequential([
        Dense(dim, input_dim=input_shape, init='he_normal')
    ])

    for i in range(n_layers):
        model.add(_highway())

    model.add(Dense(n_classes))
    model.add(Activation(top_act))

    return model

def create_convhighway(n_layers=10, dim=32, input_shape=[28, 28], n_classes=10, shared=0):
    #input_shape is [n_rows, n_cols] of the input images
    act = 'relu'
    top_act = 'sigmoid' if n_classes == 1 else 'softmax'

    trans_bias = -n_layers // 10

    shared_highway = Conv2DHighway(dim, 3, 3, activation=act, transform_bias=trans_bias, border_mode='same')

    def _highway():
        if shared == 1: return shared_highway
        return Conv2DHighway(dim, 3, 3, activation=act, transform_bias=trans_bias, border_mode='same')

    nrows, ncols = input_shape
    model = Sequential([
        Reshape((1, nrows, ncols), input_shape=(nrows*ncols,)),
        Dropout(0.2),
        Convolution2D(dim, 5, 5, activation=act)
    ])

    for i in range(2):
        model.add(_highway())

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.3))

    for i in range(3):
        model.add(_highway())

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.4))

    for i in range(3):
        model.add(_highway())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(n_classes, init='he_normal', activation=top_act))

    return model