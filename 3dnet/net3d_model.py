from pathlib import Path
from tqdm import tqdm
import keras
import numpy as np
from sklearn.model_selection import train_test_split

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
    return convolution

def localization_module(input_layer, n_filters):
    conv1 = convolution_block(input_layer, n_filters)
    conv2 = convolution_block(conv1, n_filters, kernel = (1,1,1))
    return conv2

SIZE = 256

def conv_net(activation_type, depth = 5, n_base_filters = 16, dropout_rate = 0.3): 
    inputs = Input(shape=[32, SIZE, SIZE, 1], name='image')
    current_layer = inputs
    depth = 5
    n_base_filters = 16  #the number of filters in the first level
    n_segmentation_levels = 2
    n_labels = 1
    activation_type = "sigmoid"
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
            in_conv = convolution_block(current_layer, n_level_filters, strides = (2,2,2))

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
            segmentation_layers.insert(0, convolution_block(current_layer, n_filters = n_labels, kernel = (1,1,1)))

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
