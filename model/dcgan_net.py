import tensorflow as tf 
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, Flatten, Dense
from tensorflow.keras.initializers import TruncatedNormal 

std= 0.02 
mu = 0 
momentum = 0.9
kernel_shape = (5,5)
strides = (2, 2)
epsilon = 1e-5


def generator_model():
    model = tf.keras.models.Sequential()

    model.add(Dense(512 * 6 * 6, input_shape=(100, )))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
    model.add(Activation("relu"))
    model.add(Reshape((6, 6, 512)))  

    model.add(Conv2DTranspose(
        256,
        kernel_shape,
        strides=(2, 2),
        padding="same",
        kernel_initializer=TruncatedNormal(mean=mu, stddev=std)
    ))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(
        128,
        kernel_shape,
        strides=(2, 2),
        padding="same",
        kernel_initializer=TruncatedNormal(mean=mu, stddev=std)
    ))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(
        64,
        kernel_shape,
        strides=(2, 2),
        padding="same",
        kernel_initializer=TruncatedNormal(mean=mu, stddev=std)
    ))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(
        3,
        kernel_shape,
        strides=(2, 2),
        padding="same",
        kernel_initializer=TruncatedNormal(mean=mu, stddev=std)
    ))
    model.add(Activation("tanh"))   

    return model


def discriminator_model():
    
    model = tf.keras.Sequential()

    model.add(Conv2D(
        filters=64,    
        kernel_size=kernel_shape,   
        strides=strides,   
        padding='same',   
        input_shape=(96, 96, 3),  
        kernel_initializer=TruncatedNormal(mean=mu, stddev=std)
    ))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(
        128,
        kernel_shape,
        strides=strides,
        padding="same",
        kernel_initializer=TruncatedNormal(mean=mu, stddev=std)
    ))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))   # 加入BN层防止模式崩塌，同时加速收敛
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(
        256,
        kernel_shape,
        strides=strides,
        padding="same",
        kernel_initializer=TruncatedNormal(mean=mu, stddev=std)
    ))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(
        512,
        kernel_shape,
        strides=(2, 2),
        padding="same",
        kernel_initializer=TruncatedNormal(mean=mu, stddev=std)
    ))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())  
    model.add(Dense(1))
    model.add(Activation("sigmoid"))  

    return model


def dcgan(generator_, discriminator_):

    dcgan_ = tf.keras.Sequential()
    dcgan_.add(generator_)
    discriminator_.trainable = False  # 初始时判别器不可被训练
    dcgan_.add(discriminator_)
    return dcgan_


