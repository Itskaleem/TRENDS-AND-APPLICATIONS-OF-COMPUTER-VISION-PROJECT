import numpy as np


def crop_divisible(img, divisor=4):
    '''
    Crops the image so that the dimensions are divisible by the a given integer.
    Note: discarded pixels are always from image end, this is designed to work only for small divisors.
    '''

    h,w = img.shape[:2]

    # get remainder, crop dimension by difference
    rem_h = h % divisor
    img = img[:h - rem_h]
    # same for w
    rem_w = w % divisor
    img = img[:,:w - rem_w]

    return img

def cv_centercrop(img, size=(224,224)):
    
    '''
    Central crop of given size of an images
    '''

    h,w, = img.shape[:2]
    
    c_h = h // 2
    c_w = w // 2
    
    r_h = size[0] // 2
    r_w = size[1] // 2
    
    crop = img[c_h-r_h:c_h+r_h,c_w-r_w:c_w+r_w]

    return crop

def cv_randomcrops(img, size=(224,224), num=5):
    
    '''
    Returns a number of random crops of a specific size within an image
    '''

    h,w = img.shape[:2]

    s_h, s_w = size
    max_h, max_w = h-s_h-1, w-s_w-1

    rand_h = np.random.randint(0,max_h,num)
    rand_w = np.random.randint(0,max_w,num)

    crops = []
    for h_i, w_i in zip(rand_h, rand_w):
        crops.append(img[h_i:h_i+s_h,w_i:w_i+s_w])

    return crops
