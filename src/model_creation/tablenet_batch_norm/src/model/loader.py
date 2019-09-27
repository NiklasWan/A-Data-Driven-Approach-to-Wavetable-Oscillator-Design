import os
import numpy as np
import scipy.io.wavfile as wav_read
import glob


class AudioFileLoader():
    def __init__(self, file_path):
        
        self.x = self.load_training_data(file_path)

    def decode_audio(self, file_path):
        sample_rate, data = wav_read.read(file_path)

        if data.ndim != 1:
            data = data[:,0]

        nsamps = data.shape[0]
        n_channels = 1

        if data.dtype == np.int16:
            data = data.astype(np.float32)
            data /= 32768.

        data = np.reshape(data, (nsamps, n_channels))

        return data
        
    def load_training_data(self, file_path):
        fps = glob.glob(os.path.join(file_path, '*'))
        if len(fps) == 0:
            raise Exception('Did not find any audio files in specified directory')

        n = len(fps)
        X_train = np.empty(shape=(n, 4096, 1), dtype=np.float32)

        for i, path in enumerate(fps):
            X_train[i] = self.decode_audio(path)
        
        return X_train

    def __call__(self, bs):
        return self.x[np.random.choice(self.x.shape[0], replace=False, size=(bs,))]
