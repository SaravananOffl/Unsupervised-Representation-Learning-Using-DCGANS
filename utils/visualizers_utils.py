import matplotlib.pyplot as plt 
import numpy as np
from .image_utils import preprocess_image, scaleback_image, image_loader
from pathlib import Path
import tensorflow as tf
from .model_utils import true_samples_generator, fake_samples_generator, make_latent_samples

import cv2
import os
import glob
import re


def show_gan_losses(generator_loss, discriminator_loss, save_to=None):
    # utility function to plot the losses of the Discriminator and Generator
    plt.figure(dpi=300) 
    plt.plot(range(1, len(discriminator_loss)+1), discriminator_loss, label='Discriminator')
    plt.plot(range(1, len(discriminator_loss)+1), generator_loss, label='Generator')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Epochs vs Loss")
    plt.legend()
    if save_to is not None: 
        Path(save_to).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_to + "loss.png")
    plt.show()
    
def show_generated_images(generated_images, save_to=None, epoch_no=None, n=7):
    
    
    generated_images = (generated_images + 1)/2.0 #scale from [-1,1] to [0,1]
    # display the generated images
    # plot images
    plt.figure(dpi=300)
    
    for i in range(n * n):
        
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(generated_images[i])

    if save_to is not None:
        
        Path(save_to).mkdir(parents=True, exist_ok=True)
        
        if epoch_no is not None:
            plt.savefig(save_to + "eval_epoch_" + str(epoch_no+1) + '.png') 
        
        else:
            plt.savefig(save_to + "generated_images.png")
        
    plt.show()

# evaluate the discriminator, plot generated images, save generator model
def performance_summarizer(epoch_number, generator=None, discriminator=None, data=None, samplesize=None, num_samples=150, model_name=None ):
    
    # get real samples
    real_x, real_y = true_samples_generator(data, num_samples)
    
    # performance of the discriminator on the real dataa
    _, real_accuracy = discriminator.evaluate(real_x, real_y, verbose=0)
    
    # generate fake samples to test the discriminator
    fake_x, fake_y = fake_samples_generator(num_samples, samplesize, generator)
    
    # performance of the discriminator on the fake data
    _, fake_accuracy = discriminator.evaluate(fake_x, fake_y, verbose=0)
    

    print(f'***** Accuracy real: {real_accuracy * 100}, fake: {fake_accuracy * 100} *****')
    
    #save the generated images for future references
    show_generated_images(fake_x, save_to=f"./results/{model_name}/", epoch_no=epoch_number)
    
    #save the generator model
    generator.save(f'trained_models/{model_name}/generator')
    discriminator.save(f'trained_models/{model_name}/discriminator')

def generate_video(model_name):

    images = sorted(glob.glob(f'results/{model_name}/eval_*'), key=lambda x:float(re.findall("(\d+)",x)[0]))
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    save_to = f'results/{model_name}/outputs_evolution.avi'
    video = cv2.VideoWriter( save_to, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(image))
        
    cv2.destroyAllWindows()
    video.release()
    print(f'Epoch evolutions video saved as {save_to}')
