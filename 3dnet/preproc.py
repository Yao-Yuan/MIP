'''
Functions to preprocessing input data
- to3dpatches: cut 3d stack to smaller 3d stackes
- normalize: normalize data to 0-1
@author: Yuan
'''

import numpy as np

'''
@param:  origin - a numpy array of a 3d stack for training data
        labelImg - must have the same depth of origin, labelled data
        n_slices - preferred number of slices per output stack (please make it a even number for now)
        complete - False: only output stackes surrounding aneurysm; True: output as much as possible the whole images
@return: n_stack - number of stacks containing of n_slices slices from the images
         stacks - a list of output stacks
'''
def to3dpatches (origin, labelImg, n_slices = 32, complete = False):
    originStacks = []
    labelStacks = []
    #print(origin.shape, labelImg.shape)
    
    if complete:                                  #Directly output stacks if complete is set to true
        n_stack =int(labelImg.shape[0]/n_slices)
        output_start_index = int((labelImg.shape[0]%n_slices)/2)
        for i in range(n_stack):
            start_index = output_start_index + i * n_slices
            labelStacks.append(labelImg[start_index:start_index+n_slices, ...])   
            originStacks.append(origin[start_index:start_index+n_slices, ...]) 
        #print(n_stack)
        return n_stack, originStacks, labelStacks 
    check_zeros = []
    for i in range(labelImg.shape[0]):
        check_zeros.append(np.any(np.asarray(labelImg[i]))) #check every slice along z axis (depth) 
    non_zero_index = np.nonzero(check_zeros)
    try:
        non_zero_range = [np.amin(non_zero_index), np.amax(non_zero_index)]
    except ValueError as err:
        print("Very likely to contain zero information in the labelled data")
        return err

    n_object_slice = non_zero_range[1] - non_zero_range[0]  
    n_stack = int(n_object_slice/n_slices) + 1
    #print(non_zero_range)
    output_start_index = int(non_zero_range[0]+n_object_slice/2 - n_stack*n_slices/2)
    output_start_index = output_start_index if output_start_index > 0 else 0

    for i in range(n_stack):
        start_index = output_start_index + i * n_slices
        labelStacks.append(labelImg[start_index:start_index+n_slices, ...])   # Cut middle stackes
        originStacks.append(origin[start_index:start_index+n_slices, ...])   # Cut middle stackes
        #print(start_index)    
    return n_stack, originStacks, labelStacks

'''
@param: origin_data: input original data which will be normalize to [0,1]
        mask_data: input masks which will be normalize to 0/1
@return: x_list, y_list: normalized data of origin_data and mask_data
'''
def normalize(origin_data, mask_data):
    x_list = [(imagefile.pixel_array-imagefile.SmallestImagePixelValue)/(imagefile.LargestImagePixelValue  -imagefile.SmallestImagePixelValue) for imagefile in origin_data ]
    y_list = [(mskfile.pixel_array==1024).astype(int) for mskfile in mask_data]
    return x_list, y_list