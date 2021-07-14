from sklearn.model_selection import train_test_split
import numpy as np


def make_latent_samples(samplesize, num_samples):
    
    # generate points in the latent space
    random_points = np.random.randn( num_samples * samplesize)
    # reshape it to num_samples x samplesize 
    return random_points.reshape(num_samples, samplesize)

def true_samples_generator(data, num_samples):
    # choose random instances
    ix = np.random.randint(0, data.shape[0], num_samples)
    # retrieve selected images
    X = data[ix]
    # generate 'real' class labels (1)
    y = np.ones((num_samples, 1))
    return X, y



# generate n examples ("Fake") with class labels using the Generator model
def fake_samples_generator(num_samples, samplesize ,generator_model):
    # generate points 
    x_input = make_latent_samples(samplesize, num_samples)
    # outputs
    X = generator_model.predict(x_input)
    # generate class labels for fake - "0"
    y = np.zeros((num_samples, 1))
    return X, y


def generate_labels(size):
    # returns the labels 
    return np.ones([size, 1]), np.zeros([size, 1])


def make_trainable(model, trainable):
    # train all the layers
    for layer in model.layers:
        layer.trainable = trainable

        


