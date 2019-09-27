from keras.layers.merge import _Merge

# Taken from: https://github.com/Shaofanl/Keras-GAN/blob/master/GAN/models/layers.py

class Subtract(_Merge):
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = output-inputs[i]
        return output

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

class GradNorm(Layer):
    def __init__(self, **kwargs):
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        # Create a trainable weight variable for this layer.
        super(GradNorm, self).build(input_shapes)  # Be sure to call this somewhere!

    def call(self, inputs):
        target, wrt = inputs
        grads = K.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)

class PhaseShuffleLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(PhaseShuffleLayer, self).__init__(**kwargs)

    def build(self, input_shapes):
        # Create a trainable weight variable for this layer.
        super(PhaseShuffleLayer, self).build(input_shapes)  # Be sure to call this somewhere!

    def call(self, inputs):
        rad = 2
        
        b, x_len, nch = inputs.get_shape().as_list()
        print(x_len)
        print(nch)
        #phase = np.random.uniform(-rad, rad + 1, 1).astype(np.int32)
        phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
        pad_l = tf.maximum(phase, 0)
        pad_r = tf.maximum(-phase, 0)
        phase_start = pad_r
        x = tf.pad(inputs, [[0, 0], [pad_l, pad_r], [0, 0]], mode='reflect')

        x = x[:, phase_start:phase_start+x_len]
        x.set_shape([b, x_len, nch])

        return x

    def compute_output_shape(self, input_shapes):
        return input_shapes