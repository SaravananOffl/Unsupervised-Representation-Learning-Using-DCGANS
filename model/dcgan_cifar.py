from numpy import zeros, ones

from numpy.random import randn, randint
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dense, Dropout, BatchNormalization
from matplotlib import pyplot


#  discriminator model
def discriminator_(in_shape, kernel_shape=(3,3), d_lr=0.0002, d_beta=0.5, batchnorm=False):
    model = Sequential()
    if batchnorm:
        model.add(Conv2D(64, kernel_shape, padding='same', input_shape=in_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_shape, strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_shape, strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_shape, strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
    
    else:
        model.add(Conv2D(64, kernel_shape, padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_shape, strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_shape, strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_shape, strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=d_lr, beta_1=d_beta)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# generator model
def generator_(latent_dim, batchnorm=False):
    if batchnorm:
        model = Sequential()
        model.add(Dense(256 * 4 * 4, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    
    else:
        model = Sequential()
        model.add(Dense(256 * 4 * 4, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model

def dcgan_(generator, discriminator, lr=0.0002, beta=0.5):
    # make the discriminator not trainable -> weights wont get updated
    discriminator.trainable = False
    # connect them
    dcgan = Sequential()
    # add generator
    dcgan.add(generator)
    # add the discriminator
    dcgan.add(discriminator)
    # compile model
    opt = Adam(lr=lr, beta_1=beta)
    dcgan.compile(loss='binary_crossentropy', optimizer=opt)
    return dcgan