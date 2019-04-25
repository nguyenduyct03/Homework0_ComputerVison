import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    out = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for n in range(image.shape[2]):
                x = image[i,j,n]
                xn = x**2*0.5
                if (xn > 255):
                    xn = 255
                out[i,j,n] = xn
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    out = color.rgb2gray(image)
    ### END YOUR CODE

    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    if channel == 'R':
        out = np.copy(image)
        out[:,:,0] = 0
    
    if channel == 'G':
        out = np.copy(image)
        out[:,:,1] = 0

    if channel == 'B':
        out = np.copy(image)
        out[:,:,2] = 0
    ### END YOUR CODE

    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    out = None
    ### YOUR CODE HERE
    if channel == 'L':
        lab[:,:,1] = 0
        lab[:,:,2] = 0
    
    if channel == 'A':
        lab[:,:,0] = 0
        lab[:,:,2] = 0

    if channel == 'B':
        lab[:,:,0] = 0
        lab[:,:,1] = 0

    out = color.lab2rgb(lab)
    ### END YOUR CODE

    return out

def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    if channel == 'H':
        hsv[:,:,1] = 0
        hsv[:,:,2] = 0
    
    if channel == 'S':
        hsv[:,:,0] = 0
        hsv[:,:,2] = 0

    if channel == 'V':
        hsv[:,:,0] = 0
        hsv[:,:,1] = 0

    out = color.hsv2rgb(hsv)
    ### END YOUR CODE

    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
