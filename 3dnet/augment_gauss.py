import pydicom
import numpy as np
import os 

'''
A function addes Gaussian noise to dicom files. Processed files are stored in a directory 'gauss_augmented'
@param:  pathToInputImages - a path to a directory with dicom files
         sigma - a parameter to generate “standard normal” distribution
      
@author: Nursultan
'''

def createNoisyImages(pathToInputImages = None, sigma = 9):
    if pathToInputImages is None:
        x_path = os.getcwd() 
    else:
        x_path = pathToInputImages
        
    files = os.listdir(x_path)
    ds_list = [pydicom.read_file(os.path.join(x_path, filename)) for filename in files]
    print('Found ', len(files), ' file(s)')
    
    x = np.array([ds.pixel_array for ds in ds_list])    
    
    for i in range(len(files)):
        depth, length, width = x[i].shape
        noise = np.random.randn(length, width) * sigma
        noise = noise.astype(int)
        
        for d in range(depth):
            x[i][d] += noise
    
    directory = os.path.join(os.getcwd(), "gauss_augmented")
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(len(files)):
        file_data = files[i].split('.')
        new_file_name = file_data[0] + '_gauss' + '.' + file_data[1]
        
        new_file = ds_list[i];
        new_file.PixelData = x[i].tobytes()
        new_file.NumberOfFrames, new_file.Rows, new_file.Columns = x[i].shape
        pydicom.filewriter.write_file(os.path.join(directory, new_file_name), new_file, write_like_original=True)
        
    print('Succefully added ', len(files), ' new files with gaussian noise')
