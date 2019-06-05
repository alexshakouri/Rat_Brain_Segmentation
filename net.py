from __future__ import print_function

import numpy as np
from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import UpSampling2D
from keras.layers.core import SpatialDropout2D
from keras.models import Model

from data import channels
from data import image_cols
from data import image_rows
from data import modalities

batch_norm = True
smooth = 1.0


def dice_coef(y_true, y_pred):
    #Check for multiple regions
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #fixed matrix multiply with elementwise mutliplication
    intersection = K.sum(np.multiply(y_true_f, y_pred_f))
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def create_dice_coef(num_classes):
    #Check for multiple regions
    if num_classes == 1:
        def dice_coef_single(y_true, y_pred):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            #fixed matrix multiply with elementwise mutliplication
            intersection = K.sum(np.multiply(y_true_f, y_pred_f))
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice_coef_single
    else:
        def dice_coef_multi(y_true, y_pred):
            return 0
        return dice_coef_multi



def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)




def unet(num_classes):
    inputs = Input((image_rows, image_cols, channels * modalities))

    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    if batch_norm:
        conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)

    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    if batch_norm:
        conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    if batch_norm:
        conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    if batch_norm:
        conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    if batch_norm:
        conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)

    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    if batch_norm:
        conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    if batch_norm:
        conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Activation('relu')(conv4)

    conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    if batch_norm:
        conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Activation('relu')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
    if batch_norm:
        conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
    if batch_norm:
        conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)

    conv6 = Conv2D(256, (3, 3), padding='same')(up6)
    if batch_norm:
        conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
    if batch_norm:
        conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)

    conv7 = Conv2D(128, (3, 3), padding='same')(up7)
    if batch_norm:
        conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    if batch_norm:
        conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)

    conv8 = Conv2D(64, (3, 3), padding='same')(up8)
    if batch_norm:
        conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
    if batch_norm:
        conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)

    conv9 = Conv2D(32, (3, 3), padding='same')(up9)
    if batch_norm:
        conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
    if batch_norm:
        conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    if (num_classes == 1):
        conv10 = Conv2D(1, (1, 1), activation='softmax')(conv9)
    else:
        conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)


    model = Model(inputs=[inputs], outputs=[conv10])

    model.summary()

    return model


def double_conv_layer(x, size, dropout=0.0, batch_norm=True):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv


def ZF_UNET_224(num_classes, dropout_val=0.2, weights=None):
    if K.image_dim_ordering() == 'th':
        inputs = Input((3, 128, 128))
        axis = 1
    else:
        inputs = Input((128, 128, 3))
        axis = 3
    filters = 32

    conv_224 = double_conv_layer(inputs, filters)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16*filters)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8*filters)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val)

    conv_final = Conv2D(num_classes, (1, 1))(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="ZF_UNET_224")

    return model
