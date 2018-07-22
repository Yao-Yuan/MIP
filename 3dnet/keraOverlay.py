'''
Some functions to overload some functions in keras
@author: Yuan
'''
import os
import pydicom
import numpy as np
import math
import random
import preproc
import utils
import scipy.ndimage.filters as filters
from keras import backend as K
from functools import partial
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

'''
Box Data generator
@param: data_dir
        mask_dir
        look_up_list
        box_size - default 64
        only_valid: default False
@yield: n_box: number of boxes
        boxes: boxes as in network dimension (n, x,y,z, channel)
'''


def generate_box_data(data_dir, mask_dir, lookup_list, box_size=64, batch_size=2, only_valid=False, complete=False, smooth=1.0):
    i = 0
    while True:
        img_boxes = []
        msk_boxes = []
        for filename in lookup_list:
            try:
                sample = pydicom.read_file(os.path.join(data_dir, filename))
                mask = pydicom.read_file(os.path.join(mask_dir, filename.split('.')[0] + '.result.dcm'))
            except:
                print("Error: unable to load data!")

            sample, mask = preproc.normalizeImg(sample.pixel_array, mask.pixel_array)
            if smooth:
                sample = filters.gaussian_filter(sample, smooth)
            number, img_boxes, msk_boxes = preproc.to3dpatches(sample, mask, depth=box_size, size=box_size, complete=complete, toBoxes=True, onlyValid=only_valid)

            n_iter = int(np.ceil(number/batch_size))
            last = number%batch_size
            if last:
                for i in range(batch_size-last):                    #if not fully divided by batch, pad with the first box
                    img_boxes.append(img_boxes[0])
                    msk_boxes.append(msk_boxes[0])
            if len(img_boxes)%batch_size:
                raise ValueError('Very likely to append wrong number of box when not divided by batch size')
            for i in range(n_iter):
                yield utils.convert_data(img_boxes[i*batch_size:(i+1)*batch_size], msk_boxes[i*batch_size:(i+1)*batch_size])


'''
Generate data and mask batches to replace default data generator in keras and also preprocessing data
@parm: data_dir - directory where data files (dicom files) is stored
       mask_dir - directory where mask files (dicom files) is stored
       look_up_list - a list that contains good data to be used (corresponding to data file names)
       batch_size - specify batch size (default: 2)
@yield: x_batch: generated data files batches (numpy arrays)
        y_batch: generated mask files batches (numpy arrays)
@example to generate list from file:
    with open(normalNames) as f:
        content = f.readlines()
        normal_namelist = [x.strip() for x in content] 
'''


def generate_batch_data(data_dir, mask_dir, look_up_list, batch_size=2, scaling=2, smooth=0):
    i = 0
    while True:
        image_batch = []
        mask_batch = []
        for j in range(batch_size):
            if i == len(look_up_list):  # shuffle the data for each epoch
                i = 0
                random.shuffle(look_up_list)
            filename = look_up_list[i]
            i += 1
            try:
                sample = pydicom.read_file(os.path.join(data_dir, filename))
                mask = pydicom.read_file(os.path.join(mask_dir, filename.split('.')[0] + '.result.dcm'))
            except:
                print("Error: unable to load data!")
            image_batch.append(K.cast_to_floatx(sample.pixel_array))
            mask_batch.append(K.cast_to_floatx(mask.pixel_array))

        image_batch, mask_batch = preproc.normalize(image_batch, mask_batch)  # normalize to 0-1
        if smooth:
            image_batch = [filters.gaussian_filter(image, smooth) for image in image_batch]   #Gaussian smooth decrease noise and make scaling resonable

        image_batch = utils.padImage(image_batch, 64)  # currently pad with 0 to test network
        mask_batch = utils.padImage(mask_batch, 64)

        image_batch = [image[::scaling, ::scaling, ::scaling] for image in image_batch]
        mask_batch = [mask[::scaling, ::scaling, ::scaling] for mask in mask_batch]

        yield utils.convert_data(image_batch, mask_batch)


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return K.mean(1-dice_coef(y_true, y_pred))


# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks