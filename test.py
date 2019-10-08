from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import sys
import tensorflow as tf
import random
import warnings
warnings.filterwarnings('ignore')

from keras import backend as K
from scipy.io import savemat
from skimage.io import imsave
from skimage.transform import resize

from data import load_data
from net import unet
from net import dice_coef

from model_dilation import get_frontend
from model_dilation import get_dilation_model_unet

import pdb

weights_path = '/media/alexshakouri/TOURO Mobile USB3.03/Research/Code/brain-segmentation-master/Rat_Brain_Sementation/results/weights_multiLabel_unet_dil_1_300.h5'
train_images_path = '/media/alexshakouri/TOURO Mobile USB3.03/Research/Code/brain-segmentation-master/data/dataAll_128/'
test_images_path = '/media/alexshakouri/TOURO Mobile USB3.03/Research/Code/brain-segmentation-master/data/dataAllVal_128_testIMG/'
predictions_path = '/media/alexshakouri/TOURO Mobile USB3.03/Research/Code/brain-segmentation-master/predictions/weights_singleLabel1_matlabtest/'

num_classes = 14

imSize = 128

output_rows = 280
output_cols = 200

gpu = '0'


random.seed(1)
class_colors = [ ( random.randint(0,255),random.randint(0,255),random.randint(0,255) ) for _ in range(num_classes) ]


def predict(mean=0.0, std=1.0):
    # load and normalize data
    if mean == 0.0 and std == 1.0:
        imgs_train, _, _ = load_data(train_images_path, num_classes)
        mean = np.mean(imgs_train)
        std = np.std(imgs_train)

    imgs_test, imgs_mask_test, names_test = load_data(test_images_path, num_classes)
    
    mean = np.mean(imgs_test)
    std = np.std(imgs_test)

    original_imgs_test = imgs_test.astype(np.uint8)

    imgs_test -= mean
    imgs_test /= std

    # load model with weights
    #model = unet(num_classes) #Unet model
    #model = get_frontend(imSize,imSize, num_classes) #Dilation model
    model = get_dilation_model_unet(imSize,imSize, num_classes) #combination model

    model.load_weights(weights_path)

    # make predictions
    imgs_mask_pred = model.predict(imgs_test, verbose=1)
    # save to mat file for further processing
    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)

    matdict = {
        'pred': imgs_mask_pred,
        'image': original_imgs_test,
        'mask': imgs_mask_test,
        'name': names_test
    }
    savemat(os.path.join(predictions_path, 'predictions.mat'), matdict)
    
    # save images with segmentation and ground truth mask overlay
    for i in range(len(imgs_test)):
        pred = imgs_mask_pred[i]
        #print(original_imgs_test.shape)
        image = original_imgs_test[i]
        mask = imgs_mask_test[i]

        # segmentation mask is for the middle slice
        image_rgb = gray2rgb(image[:, :, 0])

        # prediction contour image (add all the predictions)
        pred = (np.round(pred) * 255.0).astype(np.uint8)
        # ground truth contour image (add all the masks)
        mask = (np.round(mask) * 255.0).astype(np.uint8)
                
        # combine image with contours using red for pred and blue for mask
        pred_rgb = np.array(image_rgb)
        annotation = pred_rgb[:, :, 1]
        
        #Set all the pixels with the annotation to zero and fill it in with the color
        for c in range(num_classes):
            pred_temp = pred[:,:,c]            
            mask_temp = mask[:,:,c]

            pred_temp, contours, _ = cv2.findContours(
                pred_temp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pred_temp = np.zeros(pred_temp.shape)
            cv2.drawContours(pred_temp, contours, -1, (255, 0, 0), 1)

            mask_temp, contours, _ = cv2.findContours(
                mask_temp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mask_temp = np.zeros(mask_temp.shape)
            cv2.drawContours(mask_temp, contours, -1, (255, 0, 0), 1)
  
            pred[:,:,c] = pred_temp
            mask[:,:,c] = mask_temp

            annotation[np.maximum(pred[:,:,c], mask[:,:,c]) == 255] = 0
        
        pred_rgb[:, :, 0] = pred_rgb[:, :, 1] = pred_rgb[:, :, 2] = annotation

        for c in range(num_classes):
            pred_rgb[:, :, 2] = np.maximum(pred_rgb[:, :, 2], mask[:,:,c])
            pred_rgb[: ,: ,1] = np.maximum(pred_rgb[: ,: ,1], (pred[:,:,c]/255)* class_colors[c][1]) 
            pred_rgb[:, :, 2] = np.maximum(pred_rgb[:, :, 2], (pred[:,:,c]/255)* class_colors[c][2])
            pred_rgb[:, :, 0] = np.maximum(pred_rgb[:, :, 0], (pred[:,:,c]/255)* class_colors[c][0])

        imsave(os.path.join(predictions_path,
                            names_test[i] + '.png'), pred_rgb)

    return imgs_mask_test, imgs_mask_pred, names_test

def evaluate(imgs_mask_test, imgs_mask_pred, names_test):
    test_pred = list(zip(imgs_mask_test, imgs_mask_pred))
    name_test_pred = list(zip(names_test, test_pred))
    name_test_pred.sort(key=lambda x: x[0])

    patient_ids = []
    dc_values = []

    i = 0  # start slice index
    for p in range(len(name_test_pred)):
        # get case id (names are in format <case_id>_<slice_number>)
        p_id = '_'.join(name_test_pred[p][0].split('_')[:-1])

        # if this is the last slice for the processed case
        if p + 1 >= len(name_test_pred) or p_id not in name_test_pred[p + 1][0]:
            # ground truth segmentation:
            p_slices_mask = np.array(
                [im_m[0] for im_id, im_m in name_test_pred[i:p + 1]])
            # predicted segmentation:
            p_slices_pred = np.array(
                [im_m[1] for im_id, im_m in name_test_pred[i:p + 1]])

            patient_ids.append(p_id)
            dc_values.append(dice_coefficient(p_slices_pred, p_slices_mask))
            print(p_id + ':\t' + str(dc_values[-1]))

            i = p + 1

    return dc_values, patient_ids


def dice_coefficient(prediction, ground_truth):
    prediction = np.squeeze(prediction)
    ground_truth = np.squeeze(ground_truth)
    prediction = np.round(prediction).astype(int)
    ground_truth = np.round(ground_truth).astype(int)
    
    return np.sum(prediction[ground_truth == 1]) * 2.0 / (np.sum(prediction) + np.sum(ground_truth))


def gray2rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def plot_dc(labels, values):
    y_pos = np.arange(len(labels))

    fig = plt.figure(figsize=(12, 8))
    plt.barh(y_pos, values, align='center', alpha=0.5)
    plt.yticks(y_pos, labels)
    plt.xticks(np.arange(0.5, 1.0, 0.05))
    plt.xlabel('Dice coefficient', fontsize='x-large')
    plt.axes().xaxis.grid(color='black', linestyle='-', linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0.5, 1.0])
    plt.tight_layout()
    axes.axvline(np.mean(values), color='green', linewidth=2)

    plt.savefig('DSC.png', bbox_inches='tight')
    plt.close(fig)

def post_process(imgs_mask, names_mask):
    #Step1: combine all the individual masks (128x128x14 to 128x128x1)
    #np.max as some of the pixel predictions overlap (this puts a higher weight on the later regions)
    imgs_mask_group = np.max(np.round(imgs_mask) * (np.arange(imgs_mask.shape[-1]) + 1),3)
    #step2: group the images into their own groups (total 12 groups)
    #assume the names are formatted: animalID_SliceNumber
    uniqNames = defaultdict(list)
    animalSliceNum = np.zeros(len(imgs_mask))
    for i in range(len(imgs_mask)):
        #seperate slice number from the animalID
        animalID = names_mask[i].split('_')[0]
        uniqNames[animalID].append(i)

        animalSliceNum[i] = names_mask[i].split('_')[1]
        

    #step3: sort the 44 images from 1-44 (from the names) and save the output
    for IDIter, imIter in uniqNames.items():
        namesID = names_mask[imIter]
        imgs_ID_mask = imgs_mask_group[imIter]
        namesSliceNum = animalSliceNum[imIter]

        sortSliceNum = np.argsort(namesSliceNum)

        sortNamesID = namesID[sortSliceNum]
        sort_imgs_mask = imgs_ID_mask[sortSliceNum, :, :]

        #Reverse the image processing
        #output is 280x200x44
        min_length = min(output_rows, output_cols)
        max_length = max(output_rows, output_cols) 

        zeroPad = (np.ceil(((max_length*(imSize/min_length)) - imSize)/2)).astype(int)

        imgs_pad_mask = np.pad(sort_imgs_mask, ((0,0),(zeroPad,zeroPad),(0,0)), 'constant')
        imgs_post_mask = resize(imgs_pad_mask, (imgs_ID_mask.shape[0], output_rows, output_cols)).astype(np.int32)
        
        imsave(os.path.join(predictions_path, IDIter + '.tif'), imgs_post_mask)

    return 0



if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    if len(sys.argv) > 1:
        gpu = sys.argv[1]
    device = '/gpu:' + gpu

    #with tf.device(device):
    imgs_mask_test, imgs_mask_pred, names_test = predict()

    imgs_mask_post = post_process(imgs_mask_pred, names_test)

    values, labels = evaluate(imgs_mask_test, imgs_mask_pred, names_test)

    print('\nAverage DSC: ' + str(np.mean(values)))

    # plot results
    plot_dc(labels, values)
