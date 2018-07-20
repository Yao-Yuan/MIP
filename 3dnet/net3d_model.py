from pathlib import Path
from tqdm import tqdm
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K

# Shortcuts to the layer we need to make it more readable.
Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
BatchNormalization = keras.layers.BatchNormalization
Conv3D = keras.layers.Conv3D
Input = keras.layers.Input
UpSampling3D = keras.layers.UpSampling3D
Dropout = keras.layers.Dropout
Add = keras.layers.Add
Concatenate = keras.layers.concatenate
AveragePooling3D = keras.layers.AveragePooling3D

def convolution_block(input_layer, n_filters, kernel = (3,3,3), activation = None, padding = 'same', strides = (1,1,1)):
    layer = Conv3D(n_filters, kernel, padding = padding, strides = strides)(input_layer)
    layer = BatchNormalization()(layer)
    if activation is None:    #??????
        return Activation('relu')(layer)
    else:
        return activation()(layer)
    
def context_module(input_layer, n_filters, dropout_rate = 0.3): #??  channels_first
    conv1 = convolution_block(input_layer = input_layer, n_filters= n_filters)
    dropout = Dropout(rate = dropout_rate)(conv1)
    conv2 = convolution_block(input_layer = dropout, n_filters = n_filters)
    return conv2

def up_sampling_module(input_layer, n_filters, size = (2,2,2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = convolution_block(up_sample, n_filters)
    #conv3dtranspose
    return convolution

def localization_module(input_layer, n_filters):
    conv1 = convolution_block(input_layer, n_filters)
    conv2 = convolution_block(conv1, n_filters, kernel = (1,1,1))
    return conv2
'''
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 1
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
'''

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    K.print_tensor(intersection, message='')
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def conv_net(size = 256, activation_type = "sigmoid", n_slices = 32, depth = 5, n_base_filters = 16, dropout_rate = 0.3): 
    inputs = Input(shape=[n_slices, size, size, 1], name='image')
    current_layer = inputs
    depth = 5
    n_base_filters = 16  #the number of filters in the first level
    n_segmentation_levels = 2
    n_labels = 1
    dropout_rate = 0.3

    level_output = list()
    # Downsampling path
    level_filter_numbers = list()  # a list of filter numbers
    for level_number in range(depth):

        n_level_filters = (2**level_number) * n_base_filters
        level_filter_numbers.append(n_level_filters)

        if current_layer is inputs:
            in_conv = convolution_block(current_layer, n_level_filters)
        else:
            in_conv = convolution_block(current_layer, n_level_filters, kernel = (3,3,3),strides = (2,2,2)) #decrease kernel due to memory limit

        context_output = context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation = Add()([in_conv, context_output])
        level_output.append(summation)
        current_layer = summation

    # Up sampling path
    segmentation_layers = list()
    for level_number in range(depth-2, -1, -1):
        up_sampling = up_sampling_module(current_layer, level_filter_numbers[level_number])
        concatenate = Concatenate([level_output[level_number], up_sampling], axis = -1)  #check axis
        localization = localization_module(concatenate, level_filter_numbers[level_number])
        current_layer = localization
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))
    output = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation = segmentation_layers[level_number]

        if output is None:
            output = segmentation
        else:
            output = Add()([output, segmentation])

        if level_number > 0:
            output = UpSampling3D(size = (2,2,2))(output)

    prediction = Activation(activation_type)(output)
    model = keras.models.Model(inputs=inputs, outputs=prediction)        
    return model, prediction

def simple_unet(DEPTH, SIZE):
    # A common building block in Conv Nets is this Conv3x3-BatchNorm-Relu combination.
    # The `same` padding is different from the U-Net paper and will lead to edge effects.
    image = Input(shape=[DEPTH, SIZE, SIZE, 1], name='image')

    x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same')(image)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Save the large scale output to concatenate with later.
    x_large = x
    x_small = AveragePooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(32, (3, 3, 3), padding='same')(x_small)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x_upsampled = UpSampling3D(size=(2, 2, 2))(x)
    x = keras.layers.concatenate([x_large, x_upsampled])

    x = Conv3D(16, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Logits is a common name to give to the pre-sigmoid or softmax values.
    # The convention comes statistics and logistic regression I believe.
    logits = Conv3D(1, (1, 1, 1), padding='same', name='logit')(x)

    # The sigmoid will ensure all our prediction as in the range [0, 1].
    # It also suggests a probabilistic interpretation of our output but we skip this for now.
    prediction = Activation('sigmoid', name='prediction')(logits)

    # In the functional API everything is tied together with a Model() instance.
    model = keras.models.Model(inputs=image, outputs=prediction)
    return model, prediction
