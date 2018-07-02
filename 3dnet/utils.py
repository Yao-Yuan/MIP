import os
import pydicom

'''
Filter image and mask files based on a text file and designated size (to be determined: not used now because of the use of 3d patches)
@param: textfile - a file that contains filenames that is good to use
        img_dir - path for images
        msk_dir - path for masks
        size - only return images & masks with this size (default 220*256*256)
@return: img_name_list: namelist of available images
         msk_name_list: namelist of available masks which is corresponding to the img_name_list

'''
def filter_image (textfile, img_dir, msk_dir, size = [220, 256, 256]):
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
            msk_name_list.append(msk_name)
        
    for filename in invalid_img_list:
        img_name_list.remove(filename) 

    return img_name_list, msk_name_list

'''
Resample images by interpolating scans with same layer (will make image deeper) - currently not used
@param: scan-input image (as numpy matrix)
        new_spacing- as an 1*3 array
@return: numpy matrix with new spacing
@author: Yuan
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
Visulize 3d images
@param: image - input image
        threshold
Not yet finished
'''
def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,0,1)
    
    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(121, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()