import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 

def return_cifar10():
    print("Loading Cifar 10 dataset .......")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    plt.figure(dpi=300)
    n = 7
    print(f'Total number of files in the dataset is {x_train.shape[0]}')
    print(f'Shape of each image is {x_train[0].shape}')
    print("*"*20)
    print("Examples from the dataset")
    print("*"*20)
    plt.title("Examples from the dataset")
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(x_train[i])
    plt.show()
    x_train = x_train.astype('float32')
    x_train = (x_train-127.5)/127.5
    print("Done loading the Cifar 10 dataset .......")
    return x_train 


def return_celeba():
    #returns the file names of the celeba dataset
    filenames = np.array(glob('../data/celeba_dataset/img_align_celeba/*.jpg'))
    
def return_lsun_mini():
    #returns the file names of the celeba dataset
    filenames = np.array(glob('../data/lsun_mini/*.jpg'))    

    

