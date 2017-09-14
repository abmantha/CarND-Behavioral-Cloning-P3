################################# IMPORTS #################################################
import argparse
import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
################################# IMPORTS #################################################


################################# ARG PARSER ##############################################
# Get training data folder and name of model-to-be via command line
parser = argparse.ArgumentParser(description="Executing behavior cloning model...")
parser.add_argument('filenames', nargs=2)
args = parser.parse_args()
image_directory = args.filenames[0]
model_name = args.filenames[1]
################################# ARG PARSER ##############################################

################################# MAIN PROGRAM ############################################
# Open CSV file with path to all training data images
lines = []
with open(image_directory + '/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    header = next(reader, None)
    for line in reader: 
        lines.append(line)

# Collect all 3 camera images (left, right, center) for each frame
# Convert images to RGB space
# Append a correction factor for each measurement 
#    -- equate to keeping the car from going to far left or right
images = []
measurements = []
for line in lines: 
    for i in range(3):     
        source_path = line[i]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = image_directory + '/IMG/' + filename
        image = cv2.imread(local_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    correction = 0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement + correction)
    measurements.append(measurement - correction)        

# Flip all images to double the size of the dataset 
# Multiply corresponding steering angles by -1
#   -- represent that the car is going in the opposite way
#   -- eliminate the bias of primarily left or primarily right curves
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements): 
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = -1.0 * measurement
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

# Create training data
X_train = np.array(images)
y_train = np.array(measurements)

# Nvidia CNN model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0,0))))
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

# Adam optimizer with MSE loss
model.compile(optimizer='adam', loss='mse')

# Fit and train the data with validation split of 0.2 and random shuffle
# Train for 3 epochs total, include all samples in each epoch
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

# Save model to file
model.save(model_name + '.h5')
