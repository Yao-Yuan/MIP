'''
Functions to preprocessing input data
@author: Yuan
'''

import numpy as np
from keras import backend as K

'''
Cut 3d stack to smaller 3d stackes
@pram:  origin - a numpy array of a 3d stack for training data
        labelImg - must have the same depth of origin, labelled data
        depth - preferred number of slices per output stack (please make it a even number for now)
        size - size of output boxes
        complete - False: only output stackes surrounding aneurysm; True: output as much as possible the whole images
        toBoxes - True: output boxes, False: output patches
@output: n_stack - number of stacks containing of n_slices slices from the images
         stacks - a list of output stacks
'''
def to3dpatches (origin, labelImg, depth = 32, size = 32, complete = False, toBoxes = True, onlyValid = False):
    originStacks = []
    labelStacks = []
    origin_boxes = []
    label_boxes = []
    
    if complete:                                  #Directly output stacks if complete is set to true
        n_stack =int(labelImg.shape[0]/depth)
        output_start_index = int((labelImg.shape[0]%depth)/2)
        output_start_index = output_start_index if output_start_index > 0 else 0

        for i in range(n_stack):
            start_index = output_start_index + i * depth
            originStacks.append(origin[start_index:start_index+depth, ...]) 
            labelStacks.append(labelImg[start_index:start_index+depth, ...])   
    
    else:
        check_zeros = []
        for i in range(labelImg.shape[0]):
            check_zeros.append(np.any(np.asarray(labelImg[i]))) #check every slice along z axis (depth)
        
        non_zero_index = np.nonzero(check_zeros)
        try:
            non_zero_range = [np.amin(non_zero_index), np.amax(non_zero_index)]
        except ValueError as err:
            print("Very likely to contain zero information in the labelled data")
            return err
        #print(non_zero_range)
        n_object_slice = non_zero_range[1] - non_zero_range[0]  
        n_stack = int(n_object_slice/depth) + 1
    #print(non_zero_range)
        output_start_index = int(non_zero_range[0]+n_object_slice/2 - n_stack*depth/2)
        output_start_index = output_start_index if output_start_index > 0 else 0
        
    
        for i in range(n_stack):
            start_index = output_start_index + i * depth
            originStacks.append(origin[start_index:start_index+depth, ...])   # Cut middle stackes
            labelStacks.append(labelImg[start_index:start_index+depth, ...])   # Cut middle stackes
        
        if(origin.shape[1]%size != 0):
            print("Err: the images cannot be divided to boxes!")
            return -1
    if toBoxes:
        n_alongaxis = int(origin.shape[1]/size)    #only consider x = y for our project
        for origin_stack, label_stack in zip(originStacks, labelStacks):
            for i in range(n_alongaxis):
                for j in range(n_alongaxis):
                    origin_box = origin_stack[:,i*size:(i+1)*size,j*size:(j+1)*size]
                    lable_box = label_stack[:,i*size:(i+1)*size,j*size:(j+1)*size]
                    if onlyValid:                                 #Only out put boxes with aneurysm
                        if np.any(lable_box):
                            origin_boxes.append(origin_box)
                            label_boxes.append(lable_box)
                    else:
                        origin_boxes.append(origin_box)
                        label_boxes.append(lable_box)

        return len(origin_boxes), origin_boxes, label_boxes
    else:
        return n_stack, originStacks, labelStacks

'''
def to3dpatches (origin, depth = 32, size = 32, toBoxes = True):
    stacks = []
    boxes = []
    n_stack =int(origin.shape[0]/depth)
    output_start_index = int((labelImg.shape[0]%depth)/2)
    output_start_index = output_start_index if output_start_index > 0 else 0

    for i in range(n_stack):
        start_index = output_start_index + i * depth
            stacks.append(origin[start_index:start_index+depth, ...]) 
    if toBoxes:
        n_alongaxis = int(origin.shape[1]/size)    #only consider x = y for our project
        for origin_stack, label_stack in zip(originStacks, labelStacks):
            for i in range(n_alongaxis):
                for j in range(n_alongaxis):
                    boxes.append(origin_stack[:,i*size:(i+1)*size,j*size:(j+1)*size])
        return len(boxes), boxes
    else:
        return n_stack, stacks
'''
'''
Normalize data to 0-1 (also transform from dicom files to pixel arrays
@param: origin_data: input original data (dicom images) which will be normalize to [0,1] depending on the threshold
        mask_data: input masks which will be normalize to 0/1
        threshold_l: lower threshold
        threshold_h: higher threshold
@return: x_list, y_list: normalized data of origin_data and mask_data (pixel_array)
'''
def normalizeDicom(origin_data, mask_data, threshold_l = -1000, threshold_h = 2000):
    x_list = [ np.clip(imagefile.pixel_array, threshold_l, threshold_h)  for imagefile in origin_data]
    x_list = [ (image_array-threshold_l)/(threshold_h-threshold_l) for image_array in x_list]
    y_list = [(mskfile.pixel_array==1024).astype(np.float32) for mskfile in mask_data]
    return x_list, y_list

'''
Normalize numpy data to 0-1
@param: origin_data: input original data which will be normalize to [0,1] depending on the threshold
        mask_data: input masks which will be normalize to 0/1
        threshold_l: lower threshold
        threshold_h: higher threshold
@return: x, y: normalized data of origin_data and mask_data (pixel_array)
'''
def normalize(origin_data, mask_data, threshold_l = -1000, threshold_h = 2000):
    x_list = [ np.clip(image, threshold_l, threshold_h)  for image in origin_data]
    x_list = [ K.cast_to_floatx((data-threshold_l)/(threshold_h-threshold_l)) for data in x_list]
    y_list = [ K.cast_to_floatx(mask==1024) for mask in mask_data]
    return x_list, y_list

def normalizeImg(image, mask, threshold_l = -1000, threshold_h = 2000):
    img = np.clip(image, threshold_l, threshold_h)
    img =  K.cast_to_floatx((img-threshold_l)/(threshold_h-threshold_l))
    msk =  K.cast_to_floatx(mask==1024)
    return img, msk