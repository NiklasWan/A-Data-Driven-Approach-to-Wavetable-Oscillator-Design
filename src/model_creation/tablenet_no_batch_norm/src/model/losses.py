from keras import backend as K

# Taken from: https://github.com/Shaofanl/Keras-GAN/blob/master/GAN/models/losses.py

def multiple_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


def mean_loss(y_true, y_pred):
    return K.mean(y_pred)