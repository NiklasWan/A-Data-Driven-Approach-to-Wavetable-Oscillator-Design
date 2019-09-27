import numpy as np
import glob
import os
import scipy.io.wavfile as wav_read
from keras.applications import inception_v3
import re
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
import keras.models
import mel_spec_create


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


def construct_custom_inception(num_classes):
    # Build model

    inception_model = inception_v3.InceptionV3(include_top=False, weights='imagenet')

    x = inception_model.output
    preds = Dense(num_classes, activation='softmax')(x)

    model = keras.models.Model(inputs=inception_model.input, outputs=preds)

    # Just make custom layers trainable
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True

    return model




if __name__ == '__main__':
    fp_train = glob.glob(os.path.join('input/train', '*'))
    fp_test = glob.glob(os.path.join('input/test', '*'))

    X_train, y_train = load_files(fp_train)

    X_test, y_test = load_files(fp_test)

    unique_classes_train = np.unique(y_train)

    num_classes_train = len(unique_classes_train)

    unique_classes_test = np.unique(y_test)

    #train data
    mel_spec_create.create_subdirs(unique_classes_train)

    mel_spec_create.save_melspectrograms(X_train, y_train)

    #test data
    mel_spec_create.create_subdirs(unique_classes_test, file_path='test')

    mel_spec_create.save_melspectrograms(X_test, y_test, file_path='test')
    exit(0)

    model = construct_custom_inception(7)

    train_datagen = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input)

    train_generator = train_datagen.flow_from_directory('./train/',
                                                        target_size=(299, 299),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        shuffle=True)

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy

    step_size_train = train_generator.n // train_generator.batch_size
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_train,
                        epochs=7)

    test_datagen = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input)

    test_generator = test_datagen.flow_from_directory('./test/',
                                                        target_size=(299, 299),
                                                        color_mode='rgb',
                                                        batch_size=16,
                                                        class_mode='categorical',
                                                        shuffle=True)

    step_size_test = test_generator.n // test_generator.batch_size

    scores = model.evaluate_generator(generator=test_generator, steps=step_size_test)
    print(scores)

    model.save('trained_inception_classifier.h5')