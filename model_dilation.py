from keras.layers import Activation, Reshape, Dropout
from keras.layers import AtrousConvolution2D, Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose
from keras.models import Sequential


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


