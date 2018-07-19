'''
Trivial methods to achieve some functionalities
@author: Yuan
'''
import os
import pydicom
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
'''
Filter image and mask files based on a text file and designated size and also filter out unnatural ones
@param: textfile - a file that contains filenames that is good to use
        img_dir - path for images
        msk_dir - path for masks
        size - only return images & masks with this size (default 220*256*256) 
        mask_threshold - only output images that has mask label pixel number within the threshold.
@return: img_name_list: namelist of available images
         msk_name_list: namelist of available masks which is corresponding to the img_name_list

'''
def filter_image (textfile, img_dir, msk_dir, size = [220, 256, 256], mask_threshold = 50000000):
    with open(textfile) as f:
        content = f.readlines()
    normal_namelist = [x.strip() for x in content]

    #Get good image names
    files = os.listdir(img_dir)
    img_name_list = []
    for filename in files:
        if filename in normal_namelist:
            img_name_list.append(filename)
    #print(img_name_list, normal_namelist)
    #Get corresponding masks names, delete from both name lists if mask does not exists
    msk_name_list = []
    invalid_img_list = []
    for filename in img_name_list:   #correspond images loaded
        img_namesplt = filename.split('.')
        msk_name = img_namesplt[0]+'.result.dcm'
        #print(msk_name)
        try:
            pydicom.read_file(os.path.join(msk_dir, msk_name))
        except:
            #print(filename + " does not have a result dicom file in the mask folder! will delete that from image lists")
            invalid_img_list.append(filename)
        else:
            if mask.pixel_array.shape == size and mask.pixel_array.sum() < mask_threshold:           
                msk_name_list.append(msk_name)
            else: 
                invalid_img_list.append(filename)
        
    for filename in invalid_img_list:
        img_name_list.remove(filename) 

    return img_name_list, msk_name_list

'''
Resample images by interpolating scans with same layer (will make image deeper) - currently not used
@param: scan-input image (as numpy matrix)
        new_spacing- as an 1*3 array
@return: numpy matrix with new spacing
'''
def resample(scan, new_spacing=[0.42,0.42,0.42]):
    # Determine current pixel spacing
    spacing = np.array([scan.SpacingBetweenSlices,  scan.PixelSpacing[0], scan.PixelSpacing[1]], dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = scan.pixel_array.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / scan.pixel_array.shape
    new_spacing = spacing / real_resize_factor
    
    print("real" , real_resize_factor)
    scan = scipy.ndimage.interpolation.zoom(scan.pixel_array, real_resize_factor, mode='nearest')
    
    return scan

'''
Pad images that need to be predicted with zeros (currently padding at the end of image). (e.g. 220*256*256 -> 256*256*256 the last 36 layer is zeros)
@param: image_list - input images (as a list numpy matrix)
        box_size - box_size that is currently use (will padding images accordingly so that we can have integer number of boxes)
@return: out - output padded arrays
'''
def padImage (image_list, box_size):
    out  = []
    for image in image_list:
        image_size = image.shape
        padding_size = (int(image_size[0]/box_size) + 1)*box_size - image_size[0]
        out.append(np.append(image, np.zeros((padding_size, image_size[1], image_size[2])), axis=0))
    return out

'''
Visulize 3d images
@param: image - input 3d image
        threshold - thresholding the 3d image
'''


def plot_3d(image1, image2=None, threshold=-300, threshold2=0.5):
    verts, faces = measure.marching_cubes_classic(image1, threshold)

    fig = plt.figure(figsize=(20, 10))
    if image2 is None:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(121, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    ax.add_collection3d(mesh)

    ax.set_xlim(0, image1.shape[0])
    ax.set_ylim(0, image1.shape[1])
    ax.set_zlim(0, image1.shape[2])

    if image2 is not None:
        ax2 = fig.add_subplot(122, projection='3d')
        verts2, faces2 = measure.marching_cubes_classic(image2, threshold2)
        mesh2 = Poly3DCollection(verts2[faces2])

        ax2.add_collection3d(mesh2)
        ax2.set_xlim(0, image2.shape[0])
        ax2.set_ylim(0, image2.shape[1])
        ax2.set_zlim(0, image2.shape[2])

    plt.show()