import os
import numpy as np
import tensorflow as tf
import nibabel as nib

#24
scan_at_t0 = os.path.join('001_2h_ICH-Measurement.nii')
#28
scan_at_t1 = os.path.join('001_24h_ICH-Measurement.nii');

img0 = nib.load(scan_at_t0)
img1 = nib.load(scan_at_t1)

print (img0.shape)
print(img1.shape)

# Get data from nibabel image object (returns numpy memmap object)
img0_data = img0.get_data()
img1_data = img1.get_data()


# Convert to numpy ndarray (dtype: uint16)
img0_data_arr = np.asarray(img0_data)
img1_data_arr = np.asarray(img1_data)

tmp = np.empty_like(img0_data_arr)

for i in range(24):
    tmp[:,:,i] = img1_data_arr[:,:,i]

new_image = nib.Nifti1Image(tmp,img1.affine)
nib.save(new_image,'001_2h_ICH-Measurement__.nii')

# print(tmp.shape)
# print(type(img0))
# print(type(img0_data))
# print(type(img0_data_arr))
# print(type(tmp))
