import keras.backend as K
import tensorflow as tf
import numpy as np

def OneDToTwoD(x):
    return K.expand_dims(x, axis=1)

def TwoDToOneD(x):
    return x[:, 0]

def PhaseShuffeling(x):
    rad = 2
        
    b, x_len, nch = x.get_shape().as_list()

    if x_len is None:
        return x

    phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode='reflect')

    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])

    return x
    