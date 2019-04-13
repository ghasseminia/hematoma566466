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
lbl_file = "./22_24h_ICH-Measurement.nii"
img= nib.load(lbl_file)
lbl = img.get_fdata()

salam = "./22_pred_image.nii"
img= nib.load(salam)
salam = img.get_fdata()
TP2=0
target=lbl
prediction= salam
minimum = min(target.shape[2],prediction.shape[2])
target2=target[:,:,0:minimum]
prediction2=prediction[:,:,0:minimum]
img_pred1 = nib.Nifti1Image(target2,affine=np.eye(4) )
img_pred2 = nib.Nifti1Image(prediction2,affine=np.eye(4))
if (platform.system() == 'Windows'):
        nib.save(img_pred1, os.path.join('.\\','_reality_image.nii'))
        nib.save(img_pred2, os.path.join('.\\','_prediction_image.nii'))
        
prediction=salam
target=lbl
############################

def dice_score(prediction, target):
    TP=0
    TN=0
    FP=0
    FN=0
    TP1=0
    TP2=0
    minimum = min(target.shape[2],prediction.shape[2])
    Z1=np.zeros((100,1))
    Z2=np.zeros((100,1))
######
    J=0
    for z in range(0,target.shape[2]):
        k=1
        for y in range(0, target.shape[1]):
            for x in range(0, target.shape[0]):
                if (round(target[x][y][z])==1 and k==1): 
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
                if (round(prediction[x][y][z])==round(target[x][y][z]) and round(prediction[x][y][z])==1  ):  
                    TP=TP+1
                elif(round(prediction[x][y][z])!=round(target[x][y][z]) and round(prediction[x][y][z])==1  ):
                    FP=FP+1
                elif(round(prediction[x][y][z])!=round(target[x][y][z])and round(prediction[x][y][z])==0  ):
                    FN=FN+1
    dice = (2*TP)/(2*TP+FN+FP)
    return dice
print( 'dice score is')
print( dice_score(prediction, target) )

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
    ERROR=abs( (TP2-TP1) )/TP2   
    return ERROR
print( 'accuracy is')
print( 1-binary_error(prediction, target) )
