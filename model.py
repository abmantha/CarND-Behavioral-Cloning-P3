# TODO
# ======================================
# Architecture
# Augmentation techniques

import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

samples = []
with open('../data-3/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    header = next(reader, None)
    for line in reader: 
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

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

def generator(samples, batch_size=32): 
    num_samples = len(samples)
    while 1: 
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples: 
                name = '../data-3/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            augmented_images = []
            augmented_angles = []
            for image, angle in zip(images, angles): 
                augmented_images.append(image)
                augmented_angles.append(angle)
                flipped_image = cv2.flip(image, 1)
                flipped_angle = -1.0 * angle
                augmented_images.append(flipped_image)
                augmented_angles.append(flipped_angle)

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
# Consider adding a dropout layer similar to the traffic sign classification project
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
# model.fit(X_train, y_train, verbose=1, validation_split=0.2, shuffle=True, nb_epoch=3)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

# # Print keys contained in history object
# print (history_object.history.keys())

# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

model.save('model-5.h5')
