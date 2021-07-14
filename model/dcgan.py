"""
This file contains the implementation of the dcgan 
""" 

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose, Flatten, Reshape 
from tensorflow.keras.layers import Activation
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
def generator_model(input_size, # input size that was set to create the latent space (random numbers)
              kernel_std = 0.02, # std for the normal weights init mentioned in the paper
              alpha=0.2 # alpha that should be set for the LeakyReLU activation, 0.2 was mentioned in the paper
             ):
    
    generator = Sequential() 
    generator.add(Dense(4*4*1024, input_shape = (input_size, ), kernel_initializer=RandomNormal(stddev=kernel_std), use_bias=False)) # units were referred from the paper
    generator.add(Reshape(target_shape=(4, 4, 1024))) 
    generator.add(BatchNormalization()) 
    generator.add(LeakyReLU(alpha)) 

    #second block 
    generator.add(Conv2DTranspose(512, kernel_size=5, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=kernel_std),use_bias=False)) 
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha)) 

    #third block 
    generator.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=kernel_std), use_bias=False))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha))

    #fourth block 
    generator.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=kernel_std), use_bias=False))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha))

    #final block 
    generator.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=kernel_std), use_bias=False))
    generator.add(Activation('tanh'))  #final layer has tanh activation 

    return generator 


def discriminator_model(alpha=0.2, # alpha value for the leaky relu 
        kernel_std = 0.02, # std for normal weight initialization 
        ):
    # gets images as inputs and tries to tell whether it is real or fake

    discriminator = Sequential() 
    discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding='same',kernel_initializer=RandomNormal(stddev=kernel_std),    # 16x16
               input_shape=(64, 64, 3), use_bias=False))
    discriminator.add(LeakyReLU(alpha))
    discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same', 
               kernel_initializer=RandomNormal(stddev=kernel_std), use_bias=False))
    discriminator.add(BatchNormalization()) 
    discriminator.add(LeakyReLU(alpha)) 
    discriminator.add(Flatten()) 
    discriminator.add(Dense(1, kernel_initializer=RandomNormal(stddev=kernel_std), use_bias=False))
    discriminator.add(Activation('sigmoid'))
    return discriminator 

# beta_1 is the exponential decay rate for the 1st moment estimates in Adam optimizer
def DCGAN(sample_size, 
          normal_std=0.002, # standard deviation for the normal weights
          generator_lr=0.0002, # generator's learning rate
          generator_beta=0.5, # beta value for the ADAM optimizer - generator
          discriminator_lr=0.0002, #discriminator's learning rate 
          discriminator_beta=0.5, # beta value for the ADAM optimizer - discriminator
          alpha=0.2 # 
         ):
    
    # generator model 
    generator_ = generator_model(sample_size, alpha, normal_std)

    # discriminator
    discriminator_ = discriminator_model(alpha, normal_std)
    discriminator_.compile(optimizer=Adam(lr=discriminator_lr, beta_1=discriminator_beta), loss='binary_crossentropy')
    
    # GAN
    dcgan = Sequential([generator_, discriminator_])
    dcgan.compile(optimizer=Adam(lr=generator_lr, beta_1=generator_beta), loss='binary_crossentropy')
    
    return dcgan, generator_, discriminator_


                            
