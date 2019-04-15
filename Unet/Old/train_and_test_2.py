# Default to be a python3 script
import os
import platform
import pydicom
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import sys
import nibabel as nib


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

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

def get_unet0(lr=5e-5, img_row=512, img_cols=512, multigpu=1):
	''' Create the network model

		Returns:
			model, gpu_model: gpu_model is for multi-gpu training
	'''
	inputs = Input(shape=(img_row, img_cols, 1), name='net_input') # one channel
	conv1 = Conv2D(32, (3, 3), data_format='channels_last', activation='relu', padding='same')(inputs)
	conv1 = Conv2D(32, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), data_format='channels_last',activation='relu', padding='same')(pool1)
	conv2 = Conv2D(64, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), data_format='channels_last',activation='relu', padding='same')(pool2)
	conv3 = Conv2D(128, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), data_format='channels_last',activation='relu', padding='same')(pool3)
	conv4 = Conv2D(256, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), data_format='channels_last',activation='relu', padding='same')(pool4)
	conv5 = Conv2D(512, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv5)

	up6 = concatenate([Conv2DTranspose(256, (2, 2), data_format='channels_last',strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), data_format='channels_last',activation='relu', padding='same')(up6)
	conv6 = Conv2D(256, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), data_format='channels_last',strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	#up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2),
	#    padding='same')(conv6), crop(2, 0, -1)(conv3)], axis=3)
	conv7 = Conv2D(128, (3, 3), data_format='channels_last',activation='relu', padding='same')(up7)
	conv7 = Conv2D(128, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv7)

	up8 = concatenate([Conv2DTranspose(64, (2, 2), data_format='channels_last',strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), data_format='channels_last',activation='relu', padding='same')(up8)
	conv8 = Conv2D(64, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv2DTranspose(32, (2, 2), data_format='channels_last',strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), data_format='channels_last',activation='relu', padding='same')(up9)
	conv9 = Conv2D(32, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv9)

	conv10 = Conv2D(1, (1, 1), data_format='channels_last',activation='sigmoid', name='net_output')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])
	if multigpu:
		# parallelize on 2 gpus
		gpu_model = multi_gpu_model(model, 2)
		gpu_model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])
	else:
		gpu_model = None

	model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])

	return model, gpu_model


def train_model():
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)


	# Load your data here
	# X and y are numpy arrays of dimensions num_slices x height(IMG_ROWS) x width(IMG_COLS).
	# ------------------------------------------------------------------------
	# Input the size of your images here. You can play with these numbers.
	IMG_ROWS = 512
	IMG_COLS = 512

	weights_filename = './weights.h5'
	architecture_filename = './model.json'

	#X, y = load_data
	X = []
	y = []

	# pre_process
	for index in range(1,25):
		
		scan_at_t0 = os.path.join(str(index) + '_2h_ICH-Measurement' + '.nii')
		scan_at_t1 = os.path.join(str(index) + '_24h_ICH-Measurement' + '.nii')

		try:
			img0 = nib.load(scan_at_t0)
			img1 = nib.load(scan_at_t1)
		except OSError:
			continue

		img0_data = img0.get_data()
		img1_data = img1.get_data()

		img0_data_arr = np.asarray(img0_data)
		img1_data_arr = np.asarray(img1_data)

		print(img0_data_arr.shape)
		print(img1_data_arr.shape)

		tmpX = np.zeros((512,512))
		tmpy = np.zeros((512,512))
		X_slice = np.zeros((512,512))
		y_slice = np.zeros((512,512))
		y_slice_aligned = np.zeros((512,512))

		# Store info about slices with hematoma labels to help with matching
		# [index in img, number of voxels == 1]
		X_hematoma_slices = []
		y_hematoma_slices = []
		
		for i in range(min(img0.shape[2] , img1.shape[2])):
			tmpX[:,:]=img0_data_arr[:,:,i]
			tmpy[:,:]=img1_data_arr[:,:,i]
			
			print(tmpX.shape,tmpy.shape,np.count_nonzero(tmpX[:,:]),np.count_nonzero(tmpy[:,:]))
			if np.any(tmpX[:,:] == 1):
				X_hematoma_slices.append([i, np.count_nonzero(tmpX[:,:])])
			if np.any(tmpy[:,:] == 1):
				y_hematoma_slices.append([i, np.count_nonzero(tmpy[:,:])])

		index_offset = y_hematoma_slices[0][0] - X_hematoma_slices[0][0]

		#print(X_max_slice,y_max_slice,index_offset)
	
		for i in range(len(X_hematoma_slices)):
			X_slice[:,:] = img0_data_arr[:,:,X_hematoma_slices[i][0]]
			y_slice[:,:] = img1_data_arr[:,:,X_hematoma_slices[i][0] + index_offset]

			if np.any(X_slice[:,:] == 1) and np.any(y_slice[:,:] == 1):
				X_indices = np.where(X_slice[:,:] == 1)
				y_indices = np.where(y_slice[:,:] == 1)

				X_centre_row = np.mean(X_indices[0])
				X_centre_col = np.mean(X_indices[1])
				y_centre_row = np.mean(y_indices[0])
				y_centre_col = np.mean(y_indices[1])

				y_slice_aligned[:,:] = np.roll(y_slice[:,:], [int(X_centre_row - y_centre_row),int(X_centre_col - y_centre_col)], axis=(0, 1))

				if (np.count_nonzero(np.logical_and(X_slice[:,:],y_slice_aligned[:,:])) / np.count_nonzero(X_slice[:,:])) > 0.3:
					X.append(X_slice[:,:])
					y.append(y_slice_aligned[:,:])
					print(np.count_nonzero(X_slice[:,:]),np.count_nonzero(y_slice_aligned[:,:]))

	X = np.array(X, dtype=np.int16)
	y = np.array(y, dtype=np.int16)
	X = np.expand_dims(X, axis=3)
	y = np.expand_dims(y, axis=3)

	# -------------------------------------------------------------------------

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	model, gpu_model = get_unet0(lr=1e-5, img_row=IMG_ROWS, img_cols=IMG_COLS, multigpu=0)
	model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss', save_best_only=True)

	# Patience: How many epochs to wait before stopping
	# Min_delta: What is the minimum change to consider it an improvement.
	early_stopping = EarlyStopping(monitor='val_loss', patience=50, min_delta=1E-6)

	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	model.fit(X, y, batch_size=1, epochs=100, verbose=1, shuffle=True,
			  validation_split=0.3,
			  callbacks=[model_checkpoint, early_stopping])

	# serialize model to JSON
	model_json = model.to_json() # need to save to base model due to bugs with  keras multi-gpu
	with open(architecture_filename, "w") as json_file:
		json_file.write(model_json)

def predict(X, weights_filename, architecture_filename):
	# Here we just load the model
	with open(architecture_filename, "r") as json_file:
		loaded_model_json_ac = json_file.read()
	model = model_from_json(loaded_model_json_ac)
	model.load_weights(weights_filename)
	
	# Threshold to use to convert the probability map to a binary mask.
	thresh = 0.5
	# This is the output of the sigmoid
	prob_predictions = model.predict(X, verbose=1)
	# Transform the output to a binary mask.
	predictions = np.greater(prob_predictions, thresh)

	return predictions


if __name__ == '__main__':
	
	# Uncomment this line to train the network again
	train_model()

	weights_filename = './weights.h5'
	architecture_filename = './model.json'

	# pre_process of the test data

	imgrange_start = 1
	imgrange_end = 28
	
	for index in range(0,imgrange_end - imgrange_start):
		img_slices = []
		X_test = []
		scan_at_t0 = os.path.join(str(index + imgrange_start) + '_2h_ICH-Measurement' + '.nii')
		try:
			img0 = nib.load(scan_at_t0)
		except OSError:
			continue
		img0_data = img0.get_data()
		img0_data_arr = np.asarray(img0_data)

		print(img0_data_arr.shape)
		tmpX = np.zeros((512,512), dtype=np.int16)

		img_slices = [img0.shape, img0.header, img0.affine]

		num_slices = 0
		slice_indices = [-1, -1]	# [min, max]
		for i in range(img0.shape[2]):
			tmpX = np.zeros((512,512), dtype=np.int16)
			tmpX[:,:]=img0_data_arr[:,:,i]
			
			if np.any(tmpX[:,:] == 1):
				X_test.append(tmpX[:,:])
				if slice_indices[0] == -1:
					slice_indices[0] = i
					slice_indices[1] = i
				else:
					slice_indices[1] = i
				num_slices += 1
			
		X_test = np.array(X_test, dtype=np.int16)
		X_test = np.expand_dims(X_test, axis=3)

		predictions = predict(X_test, weights_filename, architecture_filename)
	
		img_pred_arr = np.zeros(img_slices[0])

		for x in range(0, img_slices[0][0]):
			for y in range(0, img_slices[0][1]):
				for z in range(slice_indices[0], slice_indices[0] + num_slices):
					if predictions[z - slice_indices[0],x,y,0] == True:
						img_pred_arr[x, y, z] = 1
		
		img_pred = nib.Nifti1Image(img_pred_arr, img_slices[2], img_slices[1])
		
		if (platform.system() == 'Windows'):
			nib.save(img_pred, os.path.join('.\\',str(index + imgrange_start) + '_pred_image.nii'))
		else:
			nib.save(img_pred, os.path.join('./',str(index + imgrange_start) + '_pred_image.nii'))


	# We need to read back the values and reconstruct the images.
	# We also need to measure the performance