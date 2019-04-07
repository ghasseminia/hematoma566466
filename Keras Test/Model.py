import os 
import nibabel as nib 
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


lbl_file1 = "./Acute/002_acute_ICH-Measurement.nii"
img1 = nib.load(lbl_file1)
lbl1 = img1.get_fdata()
lbl1 = lbl1.reshape((lbl1.shape[0],lbl1.shape[1],lbl1.shape[2]))

lbl1 = np.flipud(lbl1)
lbl1 = np.rot90(lbl1, 3,(0,1))

lbl_file2 = "./24h/002_24h_ICH-Measurement.nii"
img2 = nib.load(lbl_file2)
lbl2 = img2.get_fdata()
lbl2 = lbl2.reshape((lbl2.shape[0],lbl2.shape[1],lbl2.shape[2]))

lbl2 = np.flipud(lbl2)
lbl2 = np.rot90(lbl2, 3,(0,1))

nib.save(img1, os.path.join('.\\','niftitest.nii.gz'))

# Generate dummy data
#x_train = np.random.random((100, 100, 100, 3))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
#x_test = np.random.random((20, 100, 100, 3))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

print(lbl1[4].size)
x_train = np.array([lbl1])
y_train = np.array([lbl2])

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(512, 512, 20)))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
#score = model.evaluate(x_test, y_test, batch_size=32)

#print(score)