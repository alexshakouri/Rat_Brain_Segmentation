from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from keras import backend as K
from scipy.io import savemat
from skimage.io import imsave

from data import load_data
from net import unet
from net import dice_coef

from model_dilation import get_frontend

import pdb

#Path for server
#weights_path = 'results/weights_multi-label_dice_2_128.h5'
#train_images_path = '/home/ashakour/MRI_segmentation/data/dataAll_128/'
#test_images_path = '/home/ashakour/MRI_segmentation/data/dataAllVal_128/'
#predictions_path = './predictions/dilation_reducelr_test2'

#Path for laptop
weights_path = '/media/alexshakouri/TOURO Mobile USB3.02/Research/Code/brain-segmentation-master/Rat_Brain_Sementation/results/weights_singleLabel1_1_128.h5'
train_images_path = '/media/alexshakouri/TOURO Mobile USB3.02/Research/Code/brain-segmentation-master/data/dataAll_128/'
test_images_path = '/media/alexshakouri/TOURO Mobile USB3.02/Research/Code/brain-segmentation-master/data/dataAllVal_128/'
predictions_path = '/media/alexshakouri/TOURO Mobile USB3.02/Research/Code/brain-segmentation-master/predictions/weights_singleLabel_1/'

num_classes = 1

imSize = 128

gpu = '0'

region_mask = 5

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
    #model = unet()
    model = get_frontend(imSize,imSize, num_classes)
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
        #print(image.shape)
        image_rgb = gray2rgb(image[:, :, 0])

        # prediction contour image (add all the predictions)
        pred = (np.round(np.sum(pred[:, :, :], axis=2)) * 255.0).astype(np.uint8)
        pred, contours, _ = cv2.findContours(
            pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pred = np.zeros(pred.shape)
        cv2.drawContours(pred, contours, -1, (255, 0, 0), 1)

        # ground truth contour image (add all the masks)
        mask = (np.round(np.sum(mask[:, :, :], axis=2)) * 255.0).astype(np.uint8)
        mask, contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask = np.zeros(mask.shape)
        cv2.drawContours(mask, contours, -1, (255, 0, 0), 1)

        # combine image with contours using red for pred and blue for mask
        pred_rgb = np.array(image_rgb)
        annotation = pred_rgb[:, :, 1]
        #HERE IS THE PROBLEM
        
        annotation[np.maximum(pred, mask[:,:,0]) == 255] = 0
        pred_rgb[:, :, 0] = pred_rgb[:, :, 1] = pred_rgb[:, :, 2] = annotation
        pred_rgb[:, :, 2] = np.maximum(pred_rgb[:, :, 2], mask[:,:,0])
        pred_rgb[:, :, 0] = np.maximum(pred_rgb[:, :, 0], pred)

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


if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    if len(sys.argv) > 1:
        gpu = sys.argv[1]
    device = '/gpu:' + gpu

    #Don't have GPU working!!!!!
    #with tf.device(device):
    imgs_mask_test, imgs_mask_pred, names_test = predict()

    #check the net
    #checkDice = dice_coef(imgs_mask_pred, imgs_mask_test)
    #`print('Dice Coeff: ' + str(sess.run(checkDice)))

    values, labels = evaluate(imgs_mask_test, imgs_mask_pred, names_test)

    print('\nAverage DSC: ' + str(np.mean(values)))

    # plot results
    plot_dc(labels, values)
