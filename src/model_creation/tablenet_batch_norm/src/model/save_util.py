import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import os


def save_plots(epoch, generated_audio, file_path):
        x = np.arange(4096)
        plt.figure(figsize=(16,16))

        for i in range(1,9):
            plt.subplot(4,2, i)
            plt.plot(x, generated_audio[i-1, :,0])
            
        plt.savefig(os.path.join(file_path, f'plot_epoch_{epoch}.png'))
        plt.close()
      
        
def save_audio_samples(epoch, generated_audio, file_path):
        for i, audio in enumerate(generated_audio):  
            wav.write(os.path.join(file_path, f'audio_epoch_{epoch}_{i}.wav'), 44100, audio[:,0])
