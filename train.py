from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf



from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from data import load_data
from data import oversample
from net import dice_coef
from net import dice_coef_loss
from net import unet
from model_dilation import get_frontend
from model_dilation import add_softmax

train_images_path = '/home/ashakour/MRI_segmentation/data/dataAll_128/'
valid_images_path = '/home/ashakour/MRI_segmentation/data/dataAllVal_128/'
init_weights_path = '/home/ashakour/MRI_segmentation/semantic_segmentation/results/weights_dilation_128_NO.h5'
weights_path = '/home/ashakour/MRI_segmentation/semantic_segmentation/results/'
log_path = './logs/multi-label_dice_lrate_1'



cross_val = True

gpu = '0'

epochs = 128
batch_size = 3
base_lr = 1e-5
decay_lr = 0.01
imageDim = 128
num_classes = 14

#Check memory
def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def train():
    imgs_train, imgs_mask_train, imgs_names_train = load_data(train_images_path, num_classes)

    print('shape_train = ', imgs_train.shape)
    print('shape_names = ', imgs_names_train.shape)

    mean = np.mean(imgs_train)
    std = np.std(imgs_train)

    imgs_train -= mean
    imgs_train /= std

    imgs_valid, imgs_mask_valid, _ = load_data(valid_images_path, num_classes)

    imgs_valid -= mean
    imgs_valid /= std

    imgs_train, imgs_mask_train = oversample(imgs_train, imgs_mask_train, imgs_names_train)

    #model is unet
    #model = unet()
    #model dilated convolutions
    model = get_frontend(imageDim,imageDim, num_classes)


    #Check model size
    print("Gb = ", get_model_memory_usage(batch_size, model))

    #adding softmax is giving me a problem with the layers!!!!!
    #model = add_softmax(model)

    if os.path.exists(init_weights_path):
        model.load_weights(init_weights_path)

    optimizer = Adam(lr=base_lr, decay=decay_lr)
    model.compile(optimizer=optimizer,
                  #loss='binary_crossentropy',
                  #metrics=['accuracy', dice_coef])
		  loss=dice_coef_loss,
                  metrics=[dice_coef])

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    training_log = TensorBoard(log_dir=log_path)

    model.fit(imgs_train, imgs_mask_train,
              validation_data=(imgs_valid, imgs_mask_valid),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              callbacks=[training_log])

    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    model.save_weights(os.path.join(
        weights_path, 'weights_multi-label_lrate_1_{}.h5'.format(epochs)))


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
