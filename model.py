# TODO
# ======================================
# Architecture
# Augmentation techniques
    # Convert to YUV
    # Use all three images
    # Crop
    # Flip + Correct angles

    # Horizontal shifts and darkening ???

import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

"""
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img, ang_range, shear_range, trans_range, brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation
    '''
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows, cols, ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
        img = augment_brightness_camera_images(img)

    return img

def generate_random_samples(image, n, ang_range, shear_range, trans_range, brightness): 
    return [transform_image(image, ang_range, shear_range, trans_range, brightness) for i in range(n)]

# Grayscale image data
def grayscale(image_data): 
    return np.sum(image_data / 3, axis=3, keepdims=True)

# Normalize grayscale image data
def normalize_grayscale(image_data): 
    return (image_data - 128) / 128.0

"""

samples = []
with open('../data-5/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    header = next(reader, None)
    for line in reader: 
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.33)
print (len(train_samples))
print (len(validation_samples))

# images = []
# measurements = []
# for line in samples: 
#     for i in range(3):     
#         source_path = line[i]
#         tokens = source_path.split('/')
#         filename = tokens[-1]
#         local_path = "../data-3/IMG/" + filename
#         image = cv2.imread(local_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
#         images.append(image)
#         correction = 0.2
#     measurement = float(line[3])
#     measurements.append(measurement)
#     measurements.append(measurement + correction)
#     measurements.append(measurement - correction)        

# augmented_images = []
# augmented_measurements = []
# for image, measurement in zip(images, measurements): 
#     augmented_images.append(image)
#     augmented_measurements.append(measurement)
#     flipped_image = cv2.flip(image, 1)
#     flipped_measurement = -1.0 * measurement
#     augmented_images.append(flipped_image)
#     augmented_measurements.append(flipped_measurement)

# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)

def preprocess_image(image): 
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return new_image

def generator(samples, batch_size=32): 
    num_samples = len(samples)
    while 1: 
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples: 
                for i in range(3): 
                      source_path = batch_sample[i]
                    tokens = source_path.split('/')
                    filename = tokens[-1]
                    name = '../data-5/IMG/' + filename
                    image = cv2.imread(name)
                    image = preprocess_image(image)
                    images.append(image)
                correction = 0.1
                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                angles.append(center_angle + correction)
                angles.append(center_angle - correction)

            augmented_images = []
            augmented_angles = []
            for image, angle in zip(images, angles): 
                augmented_images.append(image)
                augmented_angles.append(angle)
                flipped_image = cv2.flip(image, 1)
                flipped_angle = -1.0 * angle
                augmented_images.append(flipped_image)
                augmented_angles.append(flipped_angle)

            # Investigate the distribution of the data 
                # 1: Not enough data for the angle sides
                # 2: Decrease the speed for corners

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Regression Model
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
# model.fit(X_train, y_train, verbose=1, validation_split=0.2, shuffle=True, nb_epoch=3)
history_object = model.fit_generator(train_generator, samples_per_epoch=(len(train_samples)//32)*32, validation_data=validation_generator, nb_val_samples=(len(validation_samples)//32)*32, nb_epoch=5, verbose=1)

# # Print keys contained in history object
# print (history_object.history.keys())

# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

model.save('model-10.h5')
