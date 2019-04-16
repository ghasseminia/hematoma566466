import os
import pandas
import nibabel as nib
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import dicom_numpy
import glob

class patient(object):
    def __init__(self, patient_id, folder_name, path_dicoms_t0, path_dicoms_t1, clinical_data, hematoma_t0_first, hematoma_t0_last, hematoma_t1_first, hematoma_t1_last):
#    def __init__(self, patient_id, folder_name, path_t0, path_t1, clinical_data, dicoms_t0, dicoms_t1, nifti_t0, nifti_t1):
        self.patient_id = patient_id # store patient's id e.g. 001
        self.folder_name = folder_name # patient's folder e.g. ICHADAPT_001_UAH_Z_B
        self.path_dicoms_t0 = path_dicoms_t0 # e.g. ./data/ICHADAPT_001_UAH_Z_B/Acute
        self.path_dicoms_t1 = path_dicoms_t1 # e.g. ./data/ICHADAPT_001_UAH_Z_B/24h
        self.clinical_data = clinical_data # row of this patient from excel sheet
        self.hematoma_t0_first= hematoma_t0_first
        self.hematoma_t0_last= hematoma_t0_last
        self.hematoma_t1_first= hematoma_t1_first
        self.hematoma_t1_last= hematoma_t1_last

 #       self.dicoms_t0=dicoms_t0
 #       self.dicoms_t1=dicoms_t1
 #       self.nifti_t0=nifti_t0
 #       self.nifti_t1=nifti_t1

def get_clinical_data(p_id, patients_ids_col, cd_arr): # get row from excel file;

    if p_id.startswith('0'): # omit zeros e.g.008->8
        p_id=p_id[1:3] #string

    if p_id.startswith('0'):
        p_id=p_id[1:2]

    if p_id.isdigit():
        #clinical_data_index=np.where(patients_ids_col==int(p_id))
        row=cd_arr.loc[int(p_id),:]
    else:
        row=cd_arr.loc[p_id,:]

    clinical_data_row=row.values
    return clinical_data_row #row of clinical data from excel

def get_patients(dicoms_path, excel_path, sheet, num_examples, train):
    clinical_data_all = pandas.read_excel(excel_path,sheet_name=sheet)
    print('--> LOADING PATIENTS (id, path to dicoms at t0, t1, clinical data) from dicom folder: ' + str(dicoms_path) + ' and excel: ' + str(excel_path))
    patients_folders = os.listdir(dicoms_path) # array of patients' folders e.g. [ICHADAPT_001_UAH_Z_B, ...]

    patients = []; # list of all patients, struct patient (id, folder name, t0, t1..)


    clinical_data_all.values
    study_number_col=clinical_data_all['Study Number'].values
    clinical_data_all.set_index('Study Number', inplace=True)

    i = 0
    for patient_folder in patients_folders: #go through all folders id data directory

        if not patient_folder.startswith('.'):
            if i < num_examples:
                i = i + 1
                patient_id = patient_folder.split("_")[1] #store patient's id e.g. 001
                print('PID: '+ str(patient_id ))
                folder_name = patient_folder #store patient's folder e.g. [ICHADAPT_001_UAH_Z_B]
                path_dicoms_t0 = dicoms_path+"/"+patient_folder+"/Acute" #path to DICOMs

                path_dicoms_t1 = dicoms_path+"/"+patient_folder+"/24h"   #path to DICOMs


                if train == 1:
                    if os.path.exists(path_dicoms_t0): # include only those patients that have both acute and 24h CT scans
                        if os.path.exists(path_dicoms_t1):
                            clinical_data=get_clinical_data(patient_id,study_number_col,clinical_data_all)
                            patients.append(patient(patient_id,folder_name,path_dicoms_t0,path_dicoms_t1,clinical_data,-1,-1,-1,-1))
                else: # for test the path t1 doesn't exist
                    if os.path.exists(path_dicoms_t0): # include only those patients that have both acute and 24h CT scans
                        clinical_data=get_clinical_data(patient_id,study_number_col,clinical_data_all)
                        patients.append(patient(patient_id,folder_name,path_dicoms_t0,path_dicoms_t1,clinical_data,-1,-1,-1,-1))

    return patients

def load_nifti(patients, nifti_path, hematoma_only):
    print('--> LOADING NIFTI from: ' + str(nifti_path) + ' hematoma_only: ' + str(hematoma_only))
    NX = []
    Ny = []
    num_patients = len(patients)

    for index in range(0,num_patients): #load 10 patients - DICOMs, nifti, clinical data

        #NIFTI get path for t0 and t1
        if os.path.exists(nifti_path+ '/' + patients[index].patient_id + '_acute_ICH-Measurement.nii'):
            nifti_at_t0 = os.path.join(nifti_path+ '/' + patients[index].patient_id + '_acute_ICH-Measurement.nii')
        else:
            nifti_at_t0 = os.path.join(nifti_path+ '/' + patients[index].patient_id + '_acute_total-Measurement.nii')

        if os.path.exists(nifti_path+ '/' + patients[index].patient_id + '_24h_ICH-Measurement.nii'):
            nifti_at_t1 = os.path.join(nifti_path+ '/' + patients[index].patient_id + '_24h_ICH-Measurement.nii')
        else:
            nifti_at_t1 = os.path.join(nifti_path+ '/' + patients[index].patient_id + '_24h_total-Measurement.nii')

        img0 = nib.load(nifti_at_t0) # load
        img1 = nib.load(nifti_at_t1)

        img0_data = img0.get_fdata() # get data
        img1_data = img1.get_fdata()

        img0_data = img0_data.reshape((img0_data.shape[0],img0_data.shape[1],img0_data.shape[2]))
        img0_data = np.flipud(img0_data)
        img0_data = np.rot90(img0_data, 3,(0,1))

        img1_data = img1_data.reshape((img1_data.shape[0],img1_data.shape[1],img1_data.shape[2]))
        img1_data = np.flipud(img1_data)
        img1_data = np.rot90(img1_data, 3,(0,1))

        img0_data_arr = np.asarray(img0_data) # to array
        img1_data_arr = np.asarray(img1_data)

        tmpNX = ((512,512))
        tmpNy = ((512,512))

        if hematoma_only == 0: # we take all slices
            tmpNX = img0_data_arr[:,:,i]
            for i in range(0,img0.shape[2]):
                NX.append(tmpNX)

            for i in range(0,img1.shape[2]):
                tmpNy = img1_data_arr[:,:,i]
                Ny.append(tmpNy)

        else:
            # IF YOU WANT TO USE ONLY SLICES WITH HEMORRHAGE

            hematoma_slices_t0 = 0
            hematoma_slices_t1 = 0
            sum_t0 = 0
            sum_t1=0
            sum_per_slice_t0 = []
            sum_per_slice_t1 = []

            for i in range(0, img0.shape[2]):
                tmpNX = img0_data_arr[:,:,i]
                if np.any(tmpNX == 1):
                    NX.append(tmpNX)
                    sum_t0 = sum_t0 + np.sum(tmpNX)
                    sum_per_slice_t0.append(sum_t0)
                    hematoma_slices_t0 = hematoma_slices_t0 + 1
                    if patients[index].hematoma_t0_first == -1:
                        patients[index].hematoma_t0_first = i

            m_t0 = sum_t0/2

            j = 0
            while m_t0 > sum_per_slice_t0[j]:
                j = j + 1

            patients[index].hematoma_t0_last = patients[index].hematoma_t0_first + hematoma_slices_t0-1

            for i in range(0, img1.shape[2]):
                tmpNy = img1_data_arr[:,:,i]
                if np.any( tmpNy == 1):
                    sum_t1 = sum_t1 + np.sum(tmpNy)
                    sum_per_slice_t1.append(sum_t1)
                    hematoma_slices_t1 = hematoma_slices_t1 + 1
                    if patients[index].hematoma_t1_first == -1:
                        patients[index].hematoma_t1_first = i

            patients[index].hematoma_t1_last = patients[index].hematoma_t1_first + hematoma_slices_t1-1

            print('PID: ' + str(patients[index].patient_id))
            if hematoma_only == 1:
                print('NIFTI t0: hematoma starts: ' + str(patients[index].hematoma_t0_first) + ' hematoma ends:' + str(patients[index].hematoma_t0_last) + ' thickness: ' + str(hematoma_slices_t0))
                print('NIFTI t1: hematoma starts: ' + str(patients[index].hematoma_t1_first) + ' hematoma ends:' + str(patients[index].hematoma_t1_last) + ' thickness: ' + str(hematoma_slices_t1) + ' ORIGINAL VALUES')

            m_t1 = sum_t1/2

            k = 0
            while m_t1 > sum_per_slice_t1[k]:
                k = k + 1

            difference = j - k # shift in slices

            l = hematoma_slices_t0
            # i = patients[index].hematoma_t0_first + difference
            tmpNy = ((512,512))
            print ('NIFTI t1 SHIFTED:')

            # while l > 0:
            #     if i > -1 and i < img1.shape[2] + 1:
            #         tmpNy = img1_data_arr[:,:,i]
            #         print('i: ' + str(i))
            #     else:
            #         tmpNy = np.zeros([512,512])
            #         print('i: empty field')
            #     Ny.append(tmpNy)
            #     l = l - 1
            #     i = i + 1
            i = patients[index].hematoma_t1_first
            while l > 0:
                if i <= patients[index].hematoma_t1_last:
                    tmpNy = img1_data_arr[:,:,i]
                    print('i: ' + str(i))
                else:
                    tmpNy = np.zeros([512,512])
                    print('i: empty field')
                Ny.append(tmpNy)
                i +=1
                l -=1

    NX = np.array(NX)
    Ny = np.array(Ny)

    return NX,Ny

# def load_nifti2(patients, nifti_path, hematoma_only):
#     print('--> LOADING NIFTI from: ' + str(nifti_path) + ' hematoma_only: ' + str(hematoma_only))
#     NX = []
#     Ny = []
#     num_patients = len(patients)
#
#     for index in range(0,num_patients): #load 10 patients - DICOMs, nifti, clinical data
#
#         #NIFTI get path for t0 and t1
#         nifti_at_t0 = os.path.join(nifti_path+ '/' + patients[index].patient_id + '_acute_ICH-Measurement.nii')
#         nifti_at_t1 = os.path.join(nifti_path+ '/' + patients[index].patient_id + '_24h_ICH-Measurement.nii')
#
#         img0 = nib.load(nifti_at_t0) # load
#         img1 = nib.load(nifti_at_t1)
#
#         img0_data = img0.get_fdata() # get data
#         img1_data = img1.get_fdata()
#
#         img0_data = img0_data.reshape((img0_data.shape[0],img0_data.shape[1],img0_data.shape[2]))
#         img0_data = np.flipud(img0_data)
#         img0_data = np.rot90(img0_data, 3,(0,1))
#
#         img1_data = img1_data.reshape((img1_data.shape[0],img1_data.shape[1],img1_data.shape[2]))
#         img1_data = np.flipud(img1_data)
#         img1_data = np.rot90(img1_data, 3,(0,1))
#
#         img0_data_arr = np.asarray(img0_data) # to array
#         img1_data_arr = np.asarray(img1_data)
#
#
#
#
#         tmpNX = ((512,512))
#         tmpNy = ((512,512))
#
#         if hematoma_only == 0: # we take all slices
#             tmpNX = img0_data_arr[:,:,i]
#             for i in range(0,img0.shape[2]):
#                 NX.append(tmpNX)
#
#             for i in range(0,img1.shape[2]):
#                 tmpNy = img1_data_arr[:,:,i]
#                 Ny.append(tmpNy)
#
#         else:
#             # IF YOU WANT TO USE ONLY SLICES WITH HEMORRHAGE
#
#             hematoma_slices_t0=0
#             hematoma_slices_t1=0
#
#             for i in range(0, img0.shape[2]):
#                 tmpNX = img0_data_arr[:,:,i]
#                 if np.any(tmpNX == 1):
#                     NX.append(tmpNX)
#                     hematoma_slices_t0 = hematoma_slices_t0 + 1
#                     if patients[index].hematoma_t0_first == -1:
#                         patients[index].hematoma_t0_first = i
#
#             for i in range(0, img1.shape[2]):
#                 tmpNy = img1_data_arr[:,:,i]
#                 if np.any( tmpNy == 1):
#                     Ny.append(tmpNy)
#
#                     hematoma_slices_t1 = hematoma_slices_t1 + 1
#                     if patients[index].hematoma_t1_first == -1:
#                         patients[index].hematoma_t1_first = i
#
#             patients[index].hematoma_t0_last = patients[index].hematoma_t0_first + hematoma_slices_t0-1
#             patients[index].hematoma_t1_last = patients[index].hematoma_t1_first + hematoma_slices_t1-1
#
#         print('PID: ' + str(patients[index].patient_id))
#         if hematoma_only == 1:
#             print('NIFTI t0: hematoma starts: ' + str(patients[index].hematoma_t0_first) + ' hematoma ends:' + str(patients[index].hematoma_t0_last) + ' thickness: ' + str(hematoma_slices_t0))
#             print('NIFTI t1: hematoma starts: ' + str(patients[index].hematoma_t1_first) + ' hematoma ends:' + str(patients[index].hematoma_t1_last) + ' thickness: ' + str(hematoma_slices_t1))
#
#     NX = np.array(NX)
#     Ny = np.array(Ny)
#
#     return NX,Ny

# # get slices of one patient
# def get_pixels_hu(dcms_paths_t0):
#     slices = [pydicom.dcmread(s) for s in dcms_paths_t0] # s one dicom ; slices for all dicoms
#     try:
#         slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
#     except:
#         slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
#
#     print('slices')
#     print(len(slices))
#     for s in slices:
#         s.SliceThickness = slice_thickness
#
#     tmp_image = ((512,512))
#     for s in slices:
#         print('*')
#         tmp_image = s.pixel_array
#         image = np.stack(tmp_image)
#     #image = np.stack([s.pixel_array for s in slices]) #pixel arrays of all dicoms
#
#     print('shape')
#     print(image.shape)
#     # Convert to int16 (from sometimes int16),
#     # should be possible as values should always be low enough (<32k)
#     image = image.astype(np.int16)
#
#     # Set outside-of-scan pixels to 0
#     # The intercept is usually -1024, so air is approximately 0
#
#     # Convert to Hounsfield units (HU)
#     intercept = slices[0].RescaleIntercept
#
#     slope = slices[0].RescaleSlope
#     #print(intercept,slope)
#
#     if slope != 1:
#         image = slope * image.astype(np.float64)
#         image = image.astype(np.int16)
#
#     image += np.int16(intercept)
#
#     image[image == -2000] = 0
#
#     return np.array(image, dtype=np.int16)

def load_dicoms(patients, dicoms_path, hematoma_only, Ny):

    DX = []
    Dy = []
    print('--> LOADING DICOMS from: ' + str(dicoms_path) + ' hematoma_only: ' + str(hematoma_only))

    for index in range(0,len(patients)): #load 10 patients - DICOMs, nifti, clinical data
        #print(patients[index].patient_id,patients[index].path_dicoms_t0,patients[index].path_dicoms_t1)
        dcms_path_t0_root = os.listdir(patients[index].path_dicoms_t0) # folder of one patient at t0
        dcms_path_t1_root = os.listdir(patients[index].path_dicoms_t1) # folder of one patient at t1

        if hematoma_only == 0:  # we look at all slices
            first_t0 = 1 # names of dicoms go from 1
            first_t1 = 1
            last_t0 = len(dcms_path_t0_root)
            last_t1 = len(dcms_path_t1_root)

        else: # we look at hematoma only
            first_t0 = patients[index].hematoma_t0_first + 1
            last_t0 = patients[index].hematoma_t0_last + 1 # names of dicoms go from 1
            first_t1 = patients[index].hematoma_t1_first + 1
            last_t1 = patients[index].hematoma_t1_last + 1

        dcms_paths_t0 = [] #paths of t0 acute scans
        dcms_paths_t1 = [] #paths of t1 24h scans

        for i in range(first_t0, last_t0 + 1):
            if i<10:
                idx='0'+str(i)
            else:
                idx=str(i)
            p_t0 =  glob.glob(patients[index].path_dicoms_t0+ '/IM-????-00' + idx+ '.dcm')
            dcm_path_t0 = os.path.join(p_t0[0])
            dcm=((512,512))

            if ".dcm" in dcm_path_t0.lower():  # check whether the file's DICOM
                dcms_paths_t0.append(dcm_path_t0)
                dcm=pydicom.dcmread(dcm_path_t0)
                dcm.slice_thickness = 5
                intercept = dcm.RescaleIntercept
                slope = dcm.RescaleSlope

                dcm_pa=dcm.pixel_array
                if slope != 1:
                    dcm_pa = slope * dcm_pa.astype(np.float64)
                    dcm_pa = dcm_pa.astype(np.int16)
                dcm_pa = dcm_pa + intercept
                dcm_pa[dcm_pa == -3000] = 0


                DX.append(dcm_pa)


        #load path to all dicomsfor i in range(first_t1, last_t1):
        # for i in range(first_t1, last_t1 + 1):
        #     if i<10:
        #         idx='0'+str(i)
        #     else:
        #         idx=str(i)
        #
        #     dcm_path_t1 = os.path.join(patients[index].path_dicoms_t1+ '/IM-0001-00' + idx+ '.dcm')
        #
        #     if ".dcm" in dcm_path_t1.lower():  # check whether the file's DICOM
        #         dcms_paths_t1.append(dcm_path_t1)
        #         dcm=pydicom.dcmread(dcm_path_t1)
        #         dcm_pa=dcm.pixel_array
        #         Dy.append(dcm_pa)

        print('PID: ' + str(patients[index].patient_id) + ' hematoma starts t0: ' + str(patients[index].hematoma_t0_first) + ' hematoma ends t0 : ' + str(patients[index].hematoma_t0_last) + ' thickness t0: ' + str(len(dcms_paths_t0)) + ' thickness t1: ' + str(len(dcms_paths_t1)) )

    DX = np.array(DX)
    #Dy = np.array(Dy)
    Dy = Ny

    return DX,Dy


def CombineDICOMandNIFTI(NX, Ny, DX, Dy):
    X=[]
    y=[]

    tmpX = np.zeros((512,512))
    tmpy = np.zeros((512,512))

    print(NX.shape, DX.shape)
    for i in range (0, NX.shape[0]):
        #print(NX.shape[0],DX.shape[0])

        tmpX[:,:]= np.multiply(NX[i],DX[i])
        # plt.imshow(DX[i], cmap=plt.cm.gray)
        # plt.show()
        plt.imshow(tmpX, cmap=plt.cm.gray)
        plt.title('tmpX: ' + str(i))
        plt.show()
        plt.imshow(Ny[i], cmap=plt.cm.gray)
        plt.title('Ny '+ str(i))
        plt.show()
        X.append(tmpX[:,:])

    y = Ny

    return X,y
