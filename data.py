from __future__ import print_function

import numpy as np
import os

#DEBUGGER
import pdb #pdb.set_trace()

from skimage.io import imread
from skimage.transform import rescale
from skimage.transform import rotate

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

import matplotlib.image as mplimg

image_rows = 128
image_cols = 128

channels = 1    # refers to neighboring slices; if set to 3, takes previous and next slice as additional channels
modalities = 1  # refers to pre, flair and post modalities; if set to 3, uses all and if set to 1, only flair
categorical = False


def load_data(path, num_classes):
    """
    Assumes filenames in given path to be in the following format as defined in `preprocessing3D.m`:
    for images: <case_id>_<slice_number>.tif
    for masks: <case_id>_<slice_number>_mask.tif

        Args:
            path: string to the folder with images

        Returns:
            np.ndarray: array of images
            np.ndarray: array of masks
            np.chararray: array of corresponding images' filenames without extensions
    """
    images_list = os.listdir(path)
    total_count = int(len(images_list) / 2) #one for mask and one for the image
    images = np.ndarray((total_count, image_rows, image_cols, channels * modalities), dtype=np.uint8)
    masks = np.ndarray((total_count, image_rows, image_cols, num_classes), dtype=np.uint8)
    split_masks = np.ndarray((image_rows, image_cols, num_classes), dtype=np.uint8)

    #This needs to be u because if it is a string then it will be bytes!!
    names = np.empty(total_count, dtype= '|U64')   
    
    #choose the region 
    region_mask = 1

    i = 0
    for image_name in images_list:
        if 'mask' in image_name:
            continue

        names[i] = image_name.split('.')[0]
        slice_number = int(names[i].split('_')[-1])
        patient_id = '_'.join(names[i].split('_')[:-1])

        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(path, image_name), as_grey=(modalities == 1))
        img_mask = imread(os.path.join(path, image_mask_name), as_grey=True)

        if channels > 1:
            img_prev = read_slice(path, patient_id, slice_number - 1)
            img_next = read_slice(path, patient_id, slice_number + 1)

            img = np.dstack((img_prev, img[..., np.newaxis], img_next))

        elif modalities == 1:
            img = np.array([img])

        img_mask = np.array([img_mask])

        #do one image for every class (single class choose a mask num)
        if(num_classes > 1):
            if (categorical):
                img_mask_int = np.squeeze(np.round(img_mask/255*14).astype(int))
                split_masks = to_categorical(img_mask_int, num_classes=num_classes+1)

            else:
                for c in range(num_classes):
                    split_masks[:,:,c] = (img_mask == np.round((c+1)*255/14))
        else:
            split_masks[:,:,0] = (img_mask == np.round(region_mask*255/14))
            #split_masks[:,:,0] = img_mask/255.

        # make the shapes line up have (1,256,256) want (256,256,1)
        img = np.squeeze(img)
        if len(img.shape) == 2:
            img = img[:,:,None]

        images[i] = img
        masks[i] = split_masks

        i += 1

    images = images.astype('float32')
    #Save space
    masks = masks.astype('int32')
    print(images.shape)
    print(masks.shape)

    return images, masks, names


def read_slice(path, patient_id, slice):
    img = np.zeros((image_rows, image_cols))
    img_name = patient_id + '_' + str(slice) + '.tif'
    img_path = os.path.join(path, img_name)

    try:
        img = imread(img_path, as_grey=(modalities == 1))
    except Exception:
        pass

    return img[..., np.newaxis]
