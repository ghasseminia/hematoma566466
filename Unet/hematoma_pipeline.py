# Main file to start training and prediction for horizontal and vertical prediction
import os
import platform
import pydicom
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import sys
import nibabel as nib
import math

# Packages from local dir
import train_and_test_horiz as horiz
import train_and_test_vert as vert

np.random.seed(197853) # for reproducibility

# import Keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import model_from_json
# custom packages
#from medo_api.core.losses import dice_coef_loss, dice_coef

import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

if __name__ == '__main__':
	
	# Uncomment this line to train the network again
	horiz.train_model_horiz()
	vert.train_model_vert()

	weights_filename_horiz = './weights_horiz.h5'
	architecture_filename_horiz = './model_horiz.json'

	weights_filename_vert = './weights_vert.h5'
	architecture_filename_vert = './model_vert.json'

	# pre_process of the test data

	imgrange_start = 1
	imgrange_end = 11
	
	for index in range(0,imgrange_end - imgrange_start):
		img_slices = []
		X_test_horiz = []
		X_test_vert = []
		scan_at_t0 = os.path.join(str(index + imgrange_start).zfill(3) + '_acute_ICH-Measurement' + '.nii')
		try:
			img0 = nib.load(scan_at_t0)
		except OSError:
			continue
		img0_data = img0.get_data()
		img0_data_arr_horiz = np.asarray(img0_data)
		img0_data_arr_vert = np.rot90(img0_data_arr_horiz, 1, (2,1))

		print(img0_data_arr_horiz.shape)


		# HORIZONTAL PREDICTION


		tmpX_horiz = np.zeros((512,512), dtype=np.int16)

		img_slices = [img0.shape, img0.header, img0.affine]

		num_slices = 0
		slice_indices = [-1, -1]	# [min, max]
		# Go through all slices in the label file
		for i in range(img0.shape[2]):
			tmpX_horiz = np.zeros((512,512), dtype=np.int16)
			tmpX_horiz[:,:]=img0_data_arr_horiz[:,:,i]
			
			# Find slices with hematoma voxels
			if np.any(tmpX_horiz[:,:] == 1):
				X_test_horiz.append(tmpX_horiz[:,:])
				if slice_indices[0] == -1:
					slice_indices[0] = i
					slice_indices[1] = i
				else:
					slice_indices[1] = i
				num_slices += 1
			
		X_test_horiz = np.array(X_test_horiz, dtype=np.int16)
		X_test_horiz = np.expand_dims(X_test_horiz, axis=3)

		predictions_horiz = horiz.predict_horiz(X_test_horiz, weights_filename_horiz, architecture_filename_horiz)
	
		img_pred_arr_horiz = np.zeros(img_slices[0])

		# Fill array with results of prediction
		for x in range(0, img_slices[0][0]):
			for y in range(0, img_slices[0][1]):
				for z in range(slice_indices[0], slice_indices[0] + num_slices):
					if predictions_horiz[z - slice_indices[0],x,y,0] == True:
						img_pred_arr_horiz[x, y, z] = 1


		# VERTICAL PREDICTION
		

		tmpX_vert = np.zeros((img0.shape[2],512), dtype=np.int16)

		img_slices = [img0_data_arr_vert.shape, img0.header, img0.affine]

		num_slices = 0
		slice_indices = [-1, -1]	# [min, max]

		# Go through all slices in the label file
		for i in range(img0.shape[0]):
			tmpX_vert = np.zeros((img0.shape[2],512), dtype=np.int16)
			tmpX_vert[:,:]=img0_data_arr_vert[i,:,:]
			
			# Find slices with hematoma voxels
			if np.any(tmpX_vert[:,:] == 1):
				X_slice = np.zeros((16,512), dtype=np.int16)
				for j in range(math.ceil((img0.shape[2] - 16) / 2), math.floor((img0.shape[2] - 16) / 2) + 16):
					X_slice[j-math.ceil((img0.shape[2] - 16) / 2),:] = tmpX_vert[j,:]
					X_test_vert.append(X_slice[:,:])
				if slice_indices[0] == -1:
					slice_indices[0] = i
					slice_indices[1] = i
				else:
					slice_indices[1] = i
				num_slices += 1
			
		X_test_vert = np.array(X_test_vert, dtype=np.int16)
		X_test_vert = np.expand_dims(X_test_vert, axis=3)

		predictions_vert = vert.predict_vert(X_test_vert, weights_filename_vert, architecture_filename_vert)
	
		img_pred_arr_vert = np.zeros(img_slices[0])

		# Fill array with results of prediction
		for x in range(0, img_slices[0][0]):
			for y in range(math.ceil((img0.shape[2] - 16) / 2), math.floor((img0.shape[2] - 16) / 2) + 16):
				for z in range(slice_indices[0], slice_indices[0] + num_slices):
					if predictions_vert[z - slice_indices[0],y-math.ceil((img0.shape[2] - 16) / 2),x,0] == True:
						img_pred_arr_vert[z, y, x] = 1

		img_pred_arr_vert = np.rot90(img_pred_arr_vert, 3, (2,1))

		img_pred_arr = np.logical_or(img_pred_arr_horiz,img_pred_arr_vert)

		img_pred = nib.Nifti1Image(img_pred_arr, img_slices[2], img_slices[1])
		
		if (platform.system() == 'Windows'):
			nib.save(img_pred, os.path.join('.\\',str(index + imgrange_start).zfill(3) + '_pred_image.nii'))
		else:
			nib.save(img_pred, os.path.join('./',str(index + imgrange_start).zfill(3) + '_pred_image.nii'))

	# We also need to measure the performance