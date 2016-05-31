import keras.backend as K

def precision(y_true, y_pred):
    y_p = K.round(y_pred)
    tp = 1.0 * K.sum(y_true * y_p)
    return tp / K.sum(y_p)

def recall(y_true, y_pred):
    y_p = K.round(y_pred)
    tp = 1.0 * K.sum(y_true * y_p)
    return tp / K.sum(y_true)

def f1(y_true, y_pred):
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    return (2 * pre * rec) / (pre + rec)
