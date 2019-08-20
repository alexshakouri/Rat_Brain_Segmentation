from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf



from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from data import load_data
from net import dice_coef
from net import dice_coef_loss
from net import unet
from model_dilation import get_frontend
from model_dilation import get_dilation_model_unet
import pdb


# dataAll_128_2 is the Veh01 in training and Veh02 in test
train_images_path = '/home/ashakour/MRI_segmentation/data/dataAll_128/'
valid_images_path = '/home/ashakour/MRI_segmentation/data/dataAllVal_128/'


#train_images_path = '/home/ashakour/MRI_segmentation/flair-segmentation/data_128/'
#valid_images_path = '/home/ashakour/MRI_segmentation/flair-segmentation/dataVal_128/'

init_weights_path = '/home/ashakour/MRI_segmentation/Rat_Brain_Sementation/results/weights_dilation_128_WHAT.h5'
weights_path = '/home/ashakour/MRI_segmentation/Rat_Brain_Sementation/results/'
log_path = '/home/ashakour/MRI_segmentation/Rat_Brain_Sementation/results/logs/test'


gpu = '0'

epochs = 1
batch_size = 32
base_lr = 2e-5
imageDim = 128
num_classes = 1 # categorical loss +1 class for background pixels

def Train_datagen(datagen1, datagen2):
    
    while True:
        image1 = datagen1.next()
        image2 = datagen2.next()
        
        #combine the two
        imageCombine = np.concatenate((image1[0], image2[0]), axis=0)
        maskCombine =  np.concatenate((image1[1], image2[1]), axis=0)

        yield imageCombine, maskCombine

def train():
    imgs_train, imgs_mask_train, imgs_names_train = load_data(train_images_path, num_classes)

    mean = np.mean(imgs_train)
    std = np.std(imgs_train)

    imgs_train -= mean
    imgs_train /= std

    imgs_valid, imgs_mask_valid, _ = load_data(valid_images_path, num_classes)

    imgs_valid -= mean
    imgs_valid /= std

    #define the model
    model = unet(num_classes)
    #model dilated convolutions
    #model = get_frontend(imageDim,imageDim, num_classes)
    #model = get_dilation_model_unet(imageDim,imageDim, num_classes)


    if os.path.exists(init_weights_path):
        model.load_weights(init_weights_path)

    optimizer = Adam(lr=base_lr)
    model.compile(optimizer=optimizer,
                  #loss='categorical_crossentropy',
                  #metrics=['accuracy', dice_coef])
		  loss=dice_coef_loss,
                  metrics=[dice_coef])

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    save_model = ModelCheckpoint(filepath=os.path.join(weights_path, "weights_test_{epoch:03d}.h5"), period=50)
    training_log = TensorBoard(log_dir=log_path)

    #Data Augmentation
    datagen = ImageDataGenerator(rotation_range = 10,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.7,1])
    datagen_flow = datagen.flow(imgs_train, imgs_mask_train, batch_size=batch_size//2)
    datagen2 = ImageDataGenerator(rotation_range = 0)
    datagen2_flow = datagen2.flow(imgs_train, imgs_mask_train, batch_size=batch_size//2)
    
    #need the len(imgs)*2 because I data augment it to twice as many
    model.fit_generator(Train_datagen(datagen_flow, datagen2_flow),
              validation_data=(imgs_valid, imgs_mask_valid),
              steps_per_epoch = (len(imgs_train)*2)//batch_size,
              epochs=epochs,
              shuffle=True,
              callbacks=[training_log, save_model])



    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    model.save_weights(os.path.join(
        weights_path, 'weights_test_1_{}.h5'.format(epochs)))


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    if len(sys.argv) > 1:
        gpu = sys.argv[1]
    device = '/gpu:' + gpu

    with tf.device(device):
        train()
