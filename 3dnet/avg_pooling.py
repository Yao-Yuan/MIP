import pydicom
import numpy as np

'''
A function returns average pooled array

- x: input array
- k_size: kernel size (default and optimal size is 3)
@author: Nursultan
'''

def getPooled(x, k_size = 3):
	depth, length, width = x.shape
	stride = 1
	padding = int((k_size - 1)/2)

	output_length = int(1 + (length - k_size + 2*padding)/stride)
	output_width = int(1 + (width - k_size + 2*padding)/stride)
	
	#output_length = int(1 + (length - k_size)/stride)
	#output_width = int(1 + (width - k_size)/stride)
	
	image_padded = np.full((depth, length + padding, width + padding), np.amax(x))
	image_padded[:, :-padding, :-padding] = x

	res = np.full((depth, output_length, output_width), np.amin(x), dtype = 'int16')

	for d in range(depth):
		for l in range(length):
			if (l + k_size <= length):
				for w in range(width):
					if (w + k_size <= width):
						res[d, l, w] = np.average(image_padded[d, l:l + k_size, w: w + k_size])
						
	return res
