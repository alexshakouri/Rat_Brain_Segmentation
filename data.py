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

import matplotlib.image as mplimg

image_rows = 128
image_cols = 128

channels = 1    # refers to neighboring slices; if set to 3, takes previous and next slice as additional channels
modalities = 1  # refers to pre, flair and post modalities; if set to 3, uses all and if set to 1, only flair


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
    total_count = int(len(images_list) / 2)
    images = np.ndarray((total_count, image_rows, image_cols, channels * modalities), dtype=np.uint8)
    masks = np.ndarray((total_count, image_rows, image_cols, num_classes), dtype=np.uint8)
    split_masks = np.ndarray((image_rows, image_cols, num_classes), dtype=np.uint8)

    #This needs to be u because if it is a string then it will be bytes!!
    names = np.empty(total_count, dtype= '|U64')   

    #choose the region 
    region_mask = 5

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
            #PUT IN THE REAL THING HERE THIS DOESN"T WORK
            for c in range(num_classes):
                split_masks[:,:,c] = (img_mask == np.round((c+1)*255/14))
        else:
            split_masks[:,:,0] = (img_mask == np.round(region_mask*255/14))

        # make the shapes line up have (1,256,256) want (256,256,1)
        img = np.squeeze(img)
        img = img[:,:,None]

        images[i] = img
        masks[i] = split_masks

        i += 1

    images = images.astype('float32')
    if (num_classes == 1):
        masks = masks[..., np.newaxis]
    
    masks = masks.astype('float32')


    return images, masks, names


def oversample(images, masks, imgs_names_train, num_classes):
    """
    Repeats 2 times every slice with nonzero mask

        Args:
            np.ndarray: array of images
            np.ndarray: array of masks

        Returns:
            np.ndarray: array of oversampled images
            np.ndarray: array of oversampled masks
    """
    images_o = []
    masks_o = []
    print(images.shape)

    
    
    #I also want to oversample the D03 images by 2
    tempNamesIndex = np.flatnonzero(np.core.defchararray.find(imgs_names_train,'D03')!=-1)   
    for i in range(len(masks)):
        if np.max(masks[i]) < 1:   
            continue

        for _ in range(2):
            images_o.append(images[i])
            masks_o.append(masks[i])

        if i in tempNamesIndex:
            for _ in range(2):
                images_o.append(images[i])
                masks_o.append(masks[i])

    images_o = np.array(images_o)
    masks_o = np.array(masks_o)

    #Data Augmentation for the images
    datagen = ImageDataGenerator(rotation_range = 10)

    images_Aug = []
    masks_Aug = []
    numImages = images_o.shape[0]    
    ImageNum = 0

    print(images_o.shape)
    for x_batch, y_batch in datagen.flow(images_o, masks_o, batch_size = numImages):
        images_Aug.append(x_batch)
        masks_Aug.append(y_batch)
        break 

    
   
    images_Aug = np.reshape(np.array(images_Aug), images_o.shape)
    masks_Aug = np.reshape(np.array(masks_Aug), masks_o.shape)

    #images_Aug = np.squeeze(images_Aug)
    #images_Aug = images_Aug[:,:,:,None]

    #masks_Aug = np.squeeze(masks_Aug)
    
    #if (num_classes == 1):
    #    masks_Aug = masks_Aug[:,:,:,None] #need the fourth for the num_classes

    #save images just to see! looks fine!
    #for i in range(len(images_Aug)):
    #    mplimg.imsave(os.path.join('./test', str(i) + '.png'), np.squeeze(images_Aug[i]))


    return np.vstack((images, images_Aug)), np.vstack((masks, masks_Aug))



def read_slice(path, patient_id, slice):
    img = np.zeros((image_rows, image_cols))
    img_name = patient_id + '_' + str(slice) + '.tif'
    img_path = os.path.join(path, img_name)

    try:
        img = imread(img_path, as_grey=(modalities == 1))
    except Exception:
        pass

    return img[..., np.newaxis]
