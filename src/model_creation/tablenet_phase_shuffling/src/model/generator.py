import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv2DTranspose, Reshape, ReLU, Lambda, BatchNormalization
from model.model_utils import OneDToTwoD, TwoDToOneD


def generator(latent_length, audio_shape, dim=64):
    s16 = dim // 16
    length, num_channels = audio_shape
    noise = Input(latent_length)

    x = noise

    x = Dense(units=(s16 * s16 * dim * 8), input_dim=latent_length)(x)
    x = Reshape((s16 * s16, dim * 8))(x)
    x = ReLU()(x)

    dim_mul = 4

    while dim_mul > 0:
        x = Lambda(OneDToTwoD)(x)
        x = Conv2DTranspose(filters=dim * dim_mul, kernel_size=(1, 25), strides=(1, 4), padding='same')(x)
        x = Lambda(TwoDToOneD)(x)
        x = ReLU()(x)
        dim_mul //= 2

    x = Lambda(OneDToTwoD)(x)
    x = Conv2DTranspose(num_channels, kernel_size=(1, 25), strides=(1, 4), padding='same')(x)
    x = Lambda(TwoDToOneD)(x)
    x = Activation('tanh')(x)

    return Model(noise, x)
