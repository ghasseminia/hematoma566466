# Default to be a python3 script
import os
import platform
import pydicom
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import sys
import nibabel as nib
import math

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


##############################equalizing number of slices to and save them to easily open them in 1 ITKsnap
lbl_file = "./046_24h_ICH-Measurement.nii"
img= nib.load(lbl_file)
target = img.get_fdata()
####################
lbl_file = "./046_pred_horiz_image.nii"
img= nib.load(lbl_file)
prediction = img.get_fdata()
####################
lbl_file = "./046_acute_ICH-Measurement.nii"
img= nib.load(lbl_file)
acute = img.get_fdata()
#################################
#acute

############################


img0_data = acute
img1_data = target

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
y_final_aligned=np.zeros((512,512,target.shape[2]))			
for i in range(min(acute.shape[2] , target.shape[2])):
    tmpX[:,:]=img0_data_arr[:,:,i]
    tmpy[:,:]=img1_data_arr[:,:,i]
			
    print(tmpX.shape,tmpy.shape,np.count_nonzero(tmpX[:,:]),np.count_nonzero(tmpy[:,:]))

			# Check whether the slices have a label and keep track of their index and hematoma voxel count
    if np.any(tmpX[:,:] == 1):
        X_hematoma_slices.append([i, np.count_nonzero(tmpX[:,:])])
    if np.any(tmpy[:,:] == 1):
        y_hematoma_slices.append([i, np.count_nonzero(tmpy[:,:])])

		# Determine the index offset to align the slices from the acute and 24h files
index_offset = y_hematoma_slices[0][0] - X_hematoma_slices[0][0]
print(index_offset)
for j in range(len(X_hematoma_slices)):
    X_slice[:,:] = img0_data_arr[:,:,X_hematoma_slices[j][0]]
    y_slice[:,:] = img1_data_arr[:,:,X_hematoma_slices[j][0] + index_offset]

			# Check that the acute and correpsonding 24h slice both have hematoma voxels
    if np.any(X_slice[:,:] == 1) and np.any(y_slice[:,:] == 1):
        X_indices = np.where(X_slice[:,:] == 1)
        y_indices = np.where(y_slice[:,:] == 1)

				# Find the mean coordinates of the hematoma in each slice
        X_centre_row = np.mean(X_indices[0])
        X_centre_col = np.mean(X_indices[1])
        y_centre_row = np.mean(y_indices[0])
        y_centre_col = np.mean(y_indices[1])

				# Create a 24h slice that is aligned to the acute hematoma slice
        y_slice_aligned[:,:] = np.roll(y_slice[:,:], [int(X_centre_row - y_centre_row),int(X_centre_col - y_centre_col)], axis=(0, 1))
        y_final_aligned[:,:,X_hematoma_slices[j][0]]= y_slice_aligned[:,:]
##########################3
minimum = min(target.shape[2],prediction.shape[2],acute.shape[2])     
target2=y_final_aligned[:,:,0:minimum]
prediction2=prediction[:,:,0:minimum]
acute2=acute[:,:,0:minimum]
img_slices = [img.shape, img.header, img.affine]
img_pred1 = nib.Nifti1Image(target2, img_slices[2], img_slices[1])
img_pred2 = nib.Nifti1Image(prediction2, img_slices[2], img_slices[1])
img_pred3 = nib.Nifti1Image(acute2, img_slices[2], img_slices[1])
####################################
if (platform.system() == 'Windows'):
        nib.save(img_pred1, os.path.join('.\\','_reality_image.nii'))
        nib.save(img_pred2, os.path.join('.\\','_prediction_image.nii'))
        nib.save(img_pred3, os.path.join('.\\','_acute_image.nii'))
##############################################
target1 = y_final_aligned
def dice_score(prediction, target1):
    TP=0
    TN=0
    FP=0
    FN=0
    TP1=0
    TP2=0
    minimum = min(target1.shape[2],prediction.shape[2])
    Z1=np.zeros((100,1))
    Z2=np.zeros((100,1))
######
    J=0
    for z in range(0,target1.shape[2]):
        k=1
        for y in range(0, target1.shape[1]):
            for x in range(0, target1.shape[0]):
                if (round(target1[x][y][z])==1 and k==1): 
                    Z1[J][0]=z
                    J=J+1
                    k=k+1
                    
#############################
    J=0
    for z in range(0,prediction.shape[2]):
        k=1
        for y in range(0, prediction.shape[1]):
            for x in range(0, prediction.shape[0]):
                if (round(prediction[x][y][z])==1 and k==1): 
                    Z2[J][0]=z
                    J=J+1
                    k=k+1
#############################                    
                   
    for x in range(0, target.shape[0]):
        for y in range(0, target.shape[1]):
            for z in range(0,minimum):
                if (round(prediction[x][y][z])==round(target1[x][y][z]) and round(prediction[x][y][z])==1  ):  
                    TP=TP+1
                elif(round(prediction[x][y][z])!=round(target1[x][y][z]) and round(prediction[x][y][z])==1  ):
                    FP=FP+1
                elif(round(prediction[x][y][z])!=round(target1[x][y][z])and round(prediction[x][y][z])==0  ):
                    FN=FN+1
    dice = (2*TP)/(2*TP+FN+FP)
    return dice
print( 'dice score is')
print( dice_score(prediction, target1) )

def binary_error(prediction, target):
    TP1=0
    TP2=0
    minimum = min(target.shape[2],prediction.shape[2])
    
    for x in range(0, target.shape[0]):
        for y in range(0, target.shape[1]):
            for z in range(0, target.shape[2]):
                if (round(target[x][y][z])==1):
                    TP2=TP2+1
    for x in range(0, prediction.shape[0]):
        for y in range(0, prediction.shape[1]):
            for z in range(0, prediction.shape[2]):               
                    if (round( prediction[x][y][z])==1):
                        TP1=TP1+1              
    Accuracy=(TP1) /TP2   
    return Accuracy
print( 'accuracy is')
print(binary_error(prediction, target) )
