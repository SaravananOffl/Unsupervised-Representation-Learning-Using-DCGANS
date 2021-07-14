from numpy import zeros, ones

from numpy.random import randn, randint
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dense, Dropout, BatchNormalization, Activation
from matplotlib import pyplot

#architecture from https://github.com/HaoRan-hash/DCGAN/blob/master/extra/network64.py

#  discriminator model
def discriminator_( kernel_shape=(5,5), strides=(2, 2), d_lr=0.0002, d_beta=0.5, batchnorm=False):
    model = Sequential()

    model.add(Conv2D(filters=64,kernel_size=kernel_shape,strides=strides,padding='same',input_shape=(64, 64, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128,(5, 5),strides=(2, 2),padding="same"))
#     model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256,(5, 5),strides=(2, 2),padding="same"))
#     model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten()) 
    model.add(Dense(1))  
    model.add(Activation("sigmoid"))
    opt = Adam(lr=d_lr, beta_1=d_beta)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# generator model
def generator_(latent_dim=(100,), batchnorm=False, kernel_shape=(5,5), strides=(2, 2)):
    model = Sequential()
    model.add(Dense(256 * 8 * 8, input_shape=(100, )))
#     model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(Activation("relu"))
    model.add(Reshape((8, 8, 256))) 
    model.add(Conv2DTranspose(128,(5, 5),strides=(2, 2),padding="same"))
#     model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(64,(5, 5),strides=(2, 2),padding="same"))
#     model.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(3,(5, 5),strides=(2, 2),padding="same"))
    model.add(Activation("tanh"))  

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
