import os
from os import system, name
def clear():  # clear the console

    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

clear()

# Default to be a python3 script

import pydicom
from pydicom.data import get_testdata_files
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import sys
import nibabel as nib
import pandas
import matplotlib.pyplot as plt
import sys
import PIL
from preprocess import get_patients,load_nifti,load_dicoms,CombineDICOMandNIFTI

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

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

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


##################################################


# =============================================================================
# ##################################################
# =============================================================================

def train_model(X,y, weights_filename, architecture_filename, IMG_ROWS, IMG_COLS):

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model, gpu_model = get_unet0(lr=1e-5, img_row=IMG_ROWS, img_cols=IMG_COLS, multigpu=0)
    model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss', save_best_only=True)

    	# Patience: How many epochs to wait before stopping
    	# Min_delta: What is the minimum change to consider it an improvement.
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=1E-3)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

# =============================================================================
    model.fit(X, y, batch_size=1, epochs=50, verbose=1, shuffle=True,
    			  validation_split=0.2,
     			  callbacks=[model_checkpoint, early_stopping])
# =============================================================================

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

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    # do not modify
    train = 1
    weights_filename = './weights.h5'
    architecture_filename = './model.json'
    IMG_ROWS = 512
    IMG_COLS = 512

    # custom
    dicoms_path = './dicoms'; # data folder with e.g. ICHADAPT_001_UAH_Z_B -> Acute/24h -> DICOMs
    nifti_path = './nifti';
    excel_path = 'clinical_data.xlsx'
    sheet = 0
    num_examples = 1 #how many files we want to load
    hematoma_only = 1  #use only DICOM and NIFTI slices with hematoma


    # Load your data here
    # X and y are numpy arrays of dimensions num_slices x height(IMG_ROWS) x width(IMG_COLS).
    # ------------------------------------------------------------------------
    # Input the size of your images here. You can play with these numbers.

    # pre_process
    patients = get_patients(dicoms_path, excel_path, sheet, num_examples, train) # 1/0 for train/test

    NX,Ny = load_nifti(patients, nifti_path, hematoma_only)

    # load_dicoms must follow after loading nifti if we want to use only hematoma slices
    DX, Dy = load_dicoms(patients, dicoms_path, hematoma_only, Ny)

    # will assign original values from DICOMs instead of 1s in X matrix
    X,y = CombineDICOMandNIFTI(NX, Ny, DX, Dy)

    X = np.expand_dims(X, axis=3) # add 4th dimension
    y = np.expand_dims(y, axis=3) # add 4th dimension

    # -------------------------------------------------------------------------
	# Uncomment this line to train the network again
    # train_model(X,y, weights_filename, architecture_filename, IMG_ROWS, IMG_COLS)
    # -------------------------------------------------------------------------

    print('-'*30)
    print('Predictions...')
    print('-'*30)

    weights_filename = './weights.h5'
    architecture_filename = './model.json'

    dicoms_path_test = './dicoms_test'; # data folder with e.g. ICHADAPT_001_UAH_Z_B -> Acute/24h -> DICOMs
    nifti_path_test = './nifti_test';
    nifti_path_results = './nifti_test_results';
    excel_path = 'clinical_data.xlsx'
    sheet = 0
    num_examples_test = 5
    hematoma_only = 1
    train = 0

	# # pre_process of the test data

	# imgrange_start = 1
	# imgrange_end = 7
	# img_slices = []
    patients_test = get_patients(dicoms_path_test, excel_path, sheet, num_examples_test, train)

    patient = []
    for index in range(0, len(patients_test)):
        patient.clear()
        patient.append(patients_test[index])
        print('patient: ' + str(patient[0].patient_id))
        NX_test,Ny_test = load_nifti(patient, nifti_path_test, hematoma_only)

        # load_dicoms must follow after loading nifti if we want to use only hematoma slices
        DX_test, Dy_test = load_dicoms(patient, dicoms_path_test, hematoma_only, Ny_test)

        # will assign original values from DICOMs instead of 1s in X matrix
        X_test,y_test = CombineDICOMandNIFTI(NX_test, Ny_test, DX_test, Dy_test)

        X_test = np.expand_dims(X_test, axis=3) # add 4th dimension
        y_test = np.expand_dims(y_test, axis=3) # add 4th dimension

        predictions = predict(X_test, weights_filename, architecture_filename)
        pred = predictions[:,:,:,0]
        print('shape: ' + str(pred.shape))
        for i in range(0,pred.shape[0]):
             plt.imshow(pred[i], cmap=plt.cm.gray)
             plt.title('i '+ str(i))
             plt.show()

        if os.path.exists(nifti_path_test+ '/' + patient[0].patient_id + '_acute_ICH-Measurement.nii'):
            nifti_at_t0 = os.path.join(nifti_path_test+ '/' + patient[0].patient_id + '_acute_ICH-Measurement.nii')
        else:
            nifti_at_t0 = os.path.join(nifti_path_test+ '/' + patient[0].patient_id + '_acute_total-Measurement.nii')

        img0 = nib.load(nifti_at_t0)
        img_slices = [img0.shape, img0.header, img0.affine]
        # print(img0.shape)
        # print(img0.header)
        # print(img0.affine)
        img_pred_arr = np.zeros(img_slices[0])

        for x in range(0, img_slices[0][0]):
            for y in range(0, img_slices[0][1]):
                for z in range(patient[0].hematoma_t0_first, patient[0].hematoma_t0_last + 1):
                    if predictions[(z - patient[0].hematoma_t0_first),x,y,0] == 0:
                        img_pred_arr[x, y, z] = 1

        img_pred = nib.Nifti1Image(img_pred_arr, img_slices[2], img_slices[1])

        final_path = os.path.join(nifti_path_results + '/')
        path_result = os.path.join(final_path + patient[0].patient_id + '_pred_image.nii')
        if not os.path.isdir(final_path):
            os.makedirs (final_path)
        nib.save(img_pred, path_result)
