from keras.models import Model
from keras.layers import Input, Dense, Activation, LeakyReLU, Conv1D, Flatten, Lambda, BatchNormalization
from model.model_utils import TwoDToOneD, PhaseShuffeling


def discriminator(input_shape, dim=64):
    slice_len , channels = input_shape

    audio = Input(input_shape)
    print("next laer")
    x = audio
    dim_mul = 2
    x = Conv1D(dim, 25, strides=4, padding='same', input_shape=input_shape)(x)
    x = LeakyReLU(alpha=0.02)(x)
    x = Lambda(PhaseShuffeling)(x)
    
    while dim_mul <= 8:
        print("next layer")
        x = Conv1D(dim * dim_mul, 25, strides=4, padding='same')(x)
        x = LeakyReLU(alpha=0.02)(x)
        x = Lambda(PhaseShuffeling)(x)

        dim_mul *= 2

    x = Flatten()(x)
    
    x = Dense(1)(x)

    return Model(audio, x)
