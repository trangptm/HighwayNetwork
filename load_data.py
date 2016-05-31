import cPickle
import gzip
import numpy
import theano
import theano.tensor as tensor

def shared_data(data_xy, borrow=True):
    data_x, data_y = data_xy

    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, tensor.cast(shared_y, 'int32')

def load(dataset):
    f = gzip.open(dataset + '.pkl.gz', 'rb')
    train, valid, test = cPickle.load(f)

    return train, valid, test

def shuffle(x, y):
    ids = numpy.arange(len(y))
    numpy.random.shuffle(ids)
    new_x = numpy.zeros(x.shape)
    new_y = numpy.zeros((len(y),))

    for i, idx in enumerate(ids):
        new_x[i] = x[idx]
        new_y[i] = y[idx]

    return new_x, new_y

def arg_passing(argv):
    # -data: dataset
    # -nlayers: number of layers of the models
    # -dim: dimension of highway layers
    # -shared: 1 if shared, 0 otherwise, 2 if both

    i = 1
    arg_dict = {'-data': 'mnist',
                '-nlayers': 10,
                '-saving': 'highway',
                '-dim': 50,
                '-shared': 0,
                '-n_rows': 28,
                '-n_cols': 28} #use for 2D image

    while i < len(argv) - 1:
        arg_dict[argv[i]] = argv[i+1]
        i += 2

    arg_dict['-dim'] = int(arg_dict['-dim'])
    arg_dict['-shared'] = int(arg_dict['-shared'])
    arg_dict['-inpShape'] = [int(arg_dict['-n_rows']), int(arg_dict['-n_cols'])]

    return arg_dict