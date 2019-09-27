import os
import librosa
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import librosa.display
import numpy as np


def create_subdirs(unique_classes, file_path='train'):

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for _class in unique_classes:
        path = os.path.join(file_path, _class)

        if not os.path.exists(path):
            os.makedirs(path)


def save_melspectrograms(X_train, y_train, file_path='train'):
    sr = 44100

    plt.figure(figsize=(4, 2))

    for i, y in enumerate(X_train):
        train_label = y_train[i]
        S = librosa.feature.melspectrogram(y=y[:, 0], sr=sr, n_mels=26, fmax=16000)
        librosa.display.specshow(librosa.power_to_db(S,
                                         ref=np.max), fmax=16000)
        plt.tight_layout()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        path = os.path.join(file_path, f'{train_label}/{train_label}_{i}.jpg')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.clf()
