'''
Some functions to overload some functions in keras
@author: Yuan
'''
import os
import pydicom
import numpy as np
import random
import preproc
import utils
from keras import backend as K
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
def generate_batch_data(data_dir, mask_dir, look_up_list, batch_size=2):
    i = 0
    image_batch = []
    mask_batch = []
    while i < 6:
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

    # print(np.array(mask_batch).shape)
        yield np.array(image_batch)[...,np.newaxis], np.array(mask_batch)[...,np.newaxis]


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    K.print_tensor(intersection, message='')
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)