from keras.models import Sequential, Model
from keras.layers import Input
from keras.optimizers import Adam
import os
import numpy as np
from model.save_util import save_audio_samples, save_plots
from model.losses import multiple_loss, mean_loss
from model.layers import GradNorm, Subtract


class GAN(object):
    def __init__(self, generator, discriminator, latent_size, **kwargs):
        self.generator = generator
        self.discriminator = discriminator
        self.gan = Sequential([generator, discriminator])
        self.latent_size = latent_size

        self.coding = self.generator.input_shape[1]

        if 'init' in kwargs:
            init = kwargs['init']
            init(self.generator)
            init(self.discriminator)
        
        generator.summary()
        discriminator.summary()
        self.gan.summary()

    def generate(self, inputs):
        return self.generator.predict(inputs)

    def build(self, **kwargs):
        opt = kwargs['opt']
       
        gen, dis, gendis = self.generator, self.discriminator, self.gan

        dis.trainable = False
        gendis.compile(optimizer=opt, loss=multiple_loss)

        shape = dis.get_input_shape_at(0)[1:]
        gen_input, real_input, interpolation = Input(shape), Input(shape), Input(shape)

        sub = Subtract()([dis(gen_input), dis(real_input)])
        norm = GradNorm()([dis(interpolation), interpolation])

        dis2batch = Model([gen_input, real_input, interpolation], [sub, norm])

        dis.trainable = True
        dis2batch.compile(optimizer=opt, loss=[mean_loss,'mse'], loss_weights=[1.0, 10.0])

        self.gen_trainer = gendis
        self.dis_trainer = dis2batch

    def fit(self, data_generator, niter=20000, nbatch=64, k=5, opt=None, save_dir='../../train/', save_iter=20):
        if opt == None:
            opt = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)

        self.build(opt=opt,)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        sample_Z = np.random.uniform(-1., 1., size=(8, self.latent_size)).astype('float32')

        for i in range(1, niter+1):
            print(f'iteration {i}')
            real_audio = data_generator(nbatch)
            Z = np.random.uniform(-1., 1., size=(nbatch, self.latent_size)).astype('float32')
            
            if (k>0 and i%(k+1) == 0) or (k<0 and i%(-k+1) != 0):
                y = np.ones((nbatch, 1)) * (-1)
                g_loss = self.gen_trainer.train_on_batch(Z, y)
                g_loss = float(g_loss)
                print('\tg_loss=%.4f'%(g_loss))

            else:
                gen_audio = self.generate(Z)
                epsilon = np.random.uniform(0, 1, size=(nbatch,1,1))
                interpolation = epsilon * real_audio + (1 - epsilon) * gen_audio

                d_loss, d_diff, d_norm = self.dis_trainer.train_on_batch([gen_audio, real_audio, interpolation], [np.ones((nbatch, 1))] * 2)
                print(f'd_loss: {d_loss}, d_diff: {d_diff}, d_norm: {d_norm}')
                
            if (i) % save_iter == 0 or i == 1:
                samples = self.generate(sample_Z)
                plot_dir = os.path.join(save_dir, 'plots')
                audio_dir = os.path.join(save_dir, 'audio')
                model_dir = os.path.join(save_dir, 'model')

                if not os.path.exists(audio_dir):
                    os.makedirs(audio_dir)
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                save_audio_samples(i, samples, audio_dir)
                save_plots(i, samples, plot_dir)

                self.gen_trainer.save(os.path.join(model_dir, 'generator_train.h5'))
                self.dis_trainer.save(os.path.join(model_dir, 'discriminator_train.h5'))

        self.gan.save(os.path.join('../../train/model/', 'trained_model_batch_norm.h5'))
        self.generator.save(os.path.join('../../train/model/', 'trained_generator_batch_norm.h5'))
        self.discriminator.save(os.path.join('../../train/model/', 'trained_discriminator_batch_norm.h5'))
