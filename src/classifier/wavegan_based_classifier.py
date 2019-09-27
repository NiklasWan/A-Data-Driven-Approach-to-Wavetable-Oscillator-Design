from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import inception_v3
import numpy as np
import scipy.io.wavfile as wav_read
import os
import re
import glob
import json


def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = BatchNormalization()(inputs)
    x = Conv2D(input_shape[0], (5, 5), padding='same', activation="relu")(x)
    x = MaxPooling2D((2, 2), (2, 2))(x)

    x = BatchNormalization()(inputs)
    x = Conv2D(input_shape[0], (5, 5), padding='same', activation="relu")(x)
    x = MaxPooling2D((2, 2), (2, 2))(x)

    x = BatchNormalization()(inputs)
    x = Conv2D(input_shape[0], (5, 5), padding='same', activation="relu")(x)
    x = MaxPooling2D((2, 2), (2, 2))(x)

    x = BatchNormalization()(inputs)
    x = Conv2D(input_shape[0], (5, 5), padding='same', activation="relu")(x)
    x = MaxPooling2D((2, 2), (2, 2))(x)

    x = Flatten()(x)
    x = BatchNormalization()(x)
    preds = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inputs, outputs=preds)


def decode_audio(file_path):
    sample_rate, data = wav_read.read(file_path)

    if data.ndim != 1:
        data = data[:, 0]

    nsamps = data.shape[0]
    n_channels = 1

    if data.dtype == np.int16:
        data = data.astype(np.float32)
        data /= 32768.

    data = np.reshape(data, (nsamps, n_channels))

    return data


def load_files(fp):
    n = len(fp)
    X_train = np.empty(shape=(n, 4096, 1), dtype=np.float32)

    y_train = []

    for i, path in enumerate(fp):
        X_train[i] = decode_audio(path)
        _path = os.path.basename(path)
        _path = re.sub(r'[_][0-9]+\.wav', "", _path)
        y_train.append(_path)

    y_train = np.array(y_train)

    return X_train, y_train


def main():
    shape = 200, 400, 3
    num_classes = 7

    fp = glob.glob(os.path.join('train', '*'))


    model = build_model(shape, num_classes)

    model.compile(optimizer=SGD(lr=1e-4), loss="categorical_crossentropy", metrics=["acc"])

    train_datagen = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input)

    train_generator = train_datagen.flow_from_directory('./train/',
                                                        target_size=(200, 400),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        shuffle=True)
    class_names = glob.glob(os.path.join('train', '*'))
    class_names = sorted(class_names)
    _class_names = []

    for i, s in enumerate(class_names):
        s = re.sub('train/', '', s)
        _class_names.append(s)

    print(_class_names)
    id_name_map = dict(zip(range(len(_class_names)), _class_names))

    with open('classes_dict.json', 'w') as fp:
        json.dump(id_name_map, fp)

    step_size_train = train_generator.n // train_generator.batch_size

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_train,
                        epochs=50)

    test_datagen = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input)

    test_generator = test_datagen.flow_from_directory('./test/',
                                                      target_size=(200, 400),
                                                      color_mode='rgb',
                                                      batch_size=16,
                                                      class_mode='categorical',
                                                      shuffle=True)

    step_size_test = test_generator.n // test_generator.batch_size

    scores = model.evaluate_generator(generator=test_generator, steps=step_size_test)
    print(scores)

    model.save('trained_custom_classifier_50.h5')


if __name__ == '__main__':
    main()