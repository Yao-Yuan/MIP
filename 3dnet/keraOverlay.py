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
def generate_batch_data(data_dir, mask_dir, look_up_list, batch_size=2, scaling=2):
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
            image_batch.append(sample.pixel_array)
            mask_batch.append(mask.pixel_array)
            image_batch, mask_batch = preproc.normalize(image_batch, mask_batch)  # normalize to 0-1

        image_batch = utils.padImage(image_batch, 64)  # currently pad with 0 to test network
        mask_batch = utils.padImage(mask_batch, 64)
        #for n in range(batch_size):
        #    output_data_batch
        image_batch = [filters.gaussian_filter(image, 1.0)[::scaling, ::scaling, ::scaling] for image in image_batch]
        mask_batch = [filters.gaussian_filter(mask, 1.0)[::scaling, ::scaling, ::scaling] for mask in mask_batch]

        #print(np.array(mask_batch).shape)
        yield np.array(image_batch)[..., np.newaxis], np.array(mask_batch)[..., np.newaxis]


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.dot(y_true_f, K.transpose(y_pred_f))
    union = K.dot(y_true_f, K.transpose(y_true_f))+K.dot(y_pred_f, K.transpose(y_pred_f))
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return K.mean(1-dice_coef(y_true, y_pred), axis=-1)


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