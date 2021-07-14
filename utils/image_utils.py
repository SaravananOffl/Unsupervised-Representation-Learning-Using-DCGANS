import numpy as np 
import matplotlib.pyplot as plt 
import scipy 
from PIL import Image

def image_loader(filename, resize_to=(32, 32), crop=False): 
   
    # takes the filename of the image as input 
    # performs cropping operation and resizes the image
   
    image = plt.imread(filename) 
    if crop:
        rows, cols = image.shape[:2] 
        crop_rows, crop_cols = 150, 150 
        row_start, cols_start = (rows - crop_rows)//2, (cols - crop_cols)//2 
        row_end, cols_end = rows - row_start, cols - row_start
        # crop the image 
        image = image[row_start:row_end, cols_start:cols_end, :] 
        image_ = np.array(Image.fromarray(image).resize(resize_to)) 
        return image_ 
    else:
        return image
 

def preprocess_image(img): 
    
    # scale the range of the image to [-1, 1]
    # mentioned in the paper 
    
    return (img.astype(np.float32) - 127.5) / 127.5

def scaleback_image(img):
    
    # scales the preprocessed images [-1, 1] to the original scale [0, 255] 
    
    return np.uint8((img+1)/2*255) 

