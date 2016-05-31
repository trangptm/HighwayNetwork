import time

from keras.optimizers import *
from keras.callbacks import *
from additional_metrics import *

import load_data
from create_model import *

# -data: dataset
# -p: p
# -log: log & model saving file
# -dim: dimension of highway layers
# -shared: 1 if shared, 0 otherwise, 2 if both

########################## LOAD DATA ###############################################
print 'Loading data...'
arg = load_data.arg_passing(sys.argv)
dataset, nlayers, saving, dim, shared = arg['-data'], arg['-nlayers'], arg['-saving'], arg['-dim'], arg['-shared']

train, valid, test = load_data.load(dataset)
log = 'log/' + saving + '.txt'

train_x, train_y = train[0], train[1]
valid_x, valid_y = load_data.shuffle(valid[0], valid[1])
test_x, test_y = load_data.shuffle(test[0], test[1])

n_classes = max(train_y)
if n_classes > 1: n_classes += 1

if n_classes == 1:
    loss = 'binary_crossentropy'
    metric = f1
    metric_str = 'f1'
else:
    loss = 'sparse_categorical_crossentropy'
    metric = 'acc'
    metric_str = 'acc'


########################## BUILD MODEL ###############################################
print 'Building model ...'
model = create_highway(nlayers, dim, train_x.shape[-1], n_classes, shared)
model.summary()

#opt = Adagrad()
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=opt,
              loss=loss, metrics=[metric])

########################## TRAINING ###############################################
print 'Training...'
start_time = time.time()

train_y = numpy.expand_dims(train_y, -1)
valid_y = numpy.expand_dims(valid_y, -1)
test_y = numpy.expand_dims(test_y, -1)

fParams = 'bestModels/' + saving + '.hdf5'
saveParams = ModelCheckpoint(fParams, monitor='val_loss', save_best_only=True)

callbacks=[saveParams]

his = model.fit(train_x, train_y,
                validation_data=(valid_x, valid_y),
                nb_epoch=20, batch_size=100,
                callbacks=callbacks)
########################## SAVE RESULTS ###############################################

# save model to json file
fModel = open('models/' + saving + '.json', 'w')
json_str = model.to_json()
fModel.write(json_str)

# load the weights for best validation result & evaluate on validation & test sets
model.load_weights(fParams)
t_loss, t_res = model.evaluate(test_x, test_y, batch_size=100)

# save training log
flog = open(log, 'w')
flog.write(str(arg) + '\n')
h = his.history
flog.write('epoch\tloss\tvloss\tv' + metric_str + '\n')
for i, x1, x2, x3 in zip(numpy.arange(len(h['loss'])), h['loss'], h['val_loss'], h['val_' + metric_str]):
    flog.write('%d\t%.4f\t%.4f\t%.4f\n' % (i, x1, x2, x3))

flog.write('Result on test set: loss: %.4f, result: %.4f\n' % (t_loss, t_res))

end_time = time.time()

flog.write('training time: %.2f' % (end_time - start_time))