from keras.layers import Activation, Reshape, Dropout, Input, BatchNormalization, concatenate
from keras.layers import AtrousConvolution2D, Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D
from keras.models import Sequential, Model
from data import modalities
from data import channels
from net import batch_norm


#
# The VGG16 keras model is taken from here:
# https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# The (caffe) structure of DilatedNet is here:
# https://github.com/fyu/dilation/blob/master/models/dilation8_pascal_voc_deploy.prototxt

def get_frontend(input_width, input_height, num_classes) -> Sequential:
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(input_width, input_height, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv1_1'))
   
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))
    model.add(ZeroPadding2D((1, 1))) # for the two maxpooling layers
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv3_3'))
    #model.add(MaxPooling2D((2, 2), strides=(1, 1)))
    #need the output to match the size of my ground truth with this pooling layer it doesn't D:

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv4_3'))

    # Compared to the original VGG16, we skip the next 2 MaxPool layers,
    # and go ahead with dilated convolutional layers instead

    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(256, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(256, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(256, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_3'))

    # Compared to the VGG16, we replace the FC layer with a convolution

#    model.add(ZeroPadding2D((12, 12)))
#    model.add(Conv2D(2048, (7, 7), dilation_rate=(4, 4), activation='relu', name='fc6'))
#    model.add(Dropout(0.5))
#    model.add(Conv2D(2048, (1, 1), activation='relu', name='fc7'))
#    model.add(Dropout(0.5))
    # Note: this layer has linear activations, not ReLU
#    model.add(Conv2D(21, (1, 1), activation='linear', name='fc-final'))

    
    #""" Append the context layers to the frontend. """
    
    #model.add(ZeroPadding2D(padding=(33, 33)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='ct_conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='ct_conv1_2'))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(64, (3, 3), dilation_rate=(2, 2), activation='relu', name='ct_conv2_1'))
    model.add(ZeroPadding2D((4, 4)))
    model.add(Conv2D(64, (3, 3), dilation_rate=(4, 4), activation='relu', name='ct_conv3_1'))
    model.add(ZeroPadding2D((8, 8)))
    model.add(Conv2D(128, (3, 3), dilation_rate=(8, 8), activation='relu', name='ct_conv4_1'))
    model.add(ZeroPadding2D((16, 16)))
    model.add(Conv2D(128, (3, 3), dilation_rate=(16, 16), activation='relu', name='ct_conv5_1'))
    
    #model.add(Conv2D(128, (3, 3), dilation_rate=(32, 32), activation='relu', name='ct_conv6_1'))
    #I need larger receptive field to take entire image    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='ct_fc1'))

    #sigmoid so I can get a probablility independent of the other classes
    model.add(Conv2D(num_classes, (1, 1), activation='sigmoid', name='ct_final'))

    # model.layers[-1].output_shape == (None, 16, 16, 21)
    model.summary()

    return model

# CITYSCAPES MODEL
def get_dilation_model_unet(image_rows, image_cols, num_classes):
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

    #conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    #if batch_norm:
    #    conv3 = BatchNormalization(axis=3)(conv3)
    #conv3 = Activation('relu')(conv3)

    #conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    #if batch_norm:
    #    conv3 = BatchNormalization(axis=3)(conv3)
    #conv3 = Activation('relu')(conv3)

    #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    #dilated convolutions
    pad1 = ZeroPadding2D((1, 1))(pool2)
    conv_dil_1 = Conv2D(256, (3, 3), activation='relu', name='ct_conv1_1')(pad1)
    pad2 = ZeroPadding2D((1, 1))(conv_dil_1)
    conv_dil_2 = Conv2D(256, (3, 3), activation='relu', name='ct_conv1_2')(pad2)
    pad3 = ZeroPadding2D((2, 2))(conv_dil_2)
    conv_dil_3 = Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='ct_conv2_1')(pad3)
    pad4 = ZeroPadding2D((4, 4))(conv_dil_3)
    conv_dil_4 = Conv2D(512, (3, 3), dilation_rate=(4, 4), activation='relu', name='ct_conv3_1')(pad4)
    pad5 = ZeroPadding2D((8, 8))(conv_dil_4)
    conv_dil_5 = Conv2D(512, (3, 3), dilation_rate=(8, 8), activation='relu', name='ct_conv4_1')(pad5)
    pad6 = ZeroPadding2D((16, 16))(conv_dil_5)
    conv_dil_6 = Conv2D(512, (3, 3), dilation_rate=(16, 16), activation='relu', name='ct_conv5_1')(pad6)

    #up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv_dil_6)
    #up7 = concatenate([up7, conv3], axis=3)

    #conv7 = Conv2D(128, (3, 3), padding='same')(up7)
    #if batch_norm:
    #    conv7 = BatchNormalization(axis=3)(conv7)
    #conv7 = Activation('relu')(conv7)

    #conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    #if batch_norm:
    #    conv7 = BatchNormalization(axis=3)(conv7)
    #conv7 = Activation('relu')(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_dil_6)
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
        conv10 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv9)


    model = Model(inputs=[inputs], outputs=[conv10])

    model.summary()
   
    return model
