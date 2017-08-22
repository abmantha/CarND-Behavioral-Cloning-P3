import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flat, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('./data/driving_log.csv', 'rb') as csvfile: 
    reader = csv.reader(csvfile)
    for line in reader: 
        print (line)
        lines.append(line)

images = []
measurements = []
for line in lines: 
    for i in range(3):     
        source_path = line[i]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = "./data/IMG" + filename
        image = csv2.imread(local_path)
        images.append(image)
    correction = 0.2
    meaurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement + correction)
    measurements.append(measurement - correction)        
    print (local_path)

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements): 
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = -1.0 * measurement
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

X_train = np.array(images)
y_train = np.array(measurements)


# Regression Model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=(70, 25), (0,0)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epochs=3)

model.save('model.h5')
s
