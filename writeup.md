**Behavioral Cloning Project** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

Woohoo :)

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model-final.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
The simulation car can be driven from within the Udacity simulator with the following command in this working directory.
```sh
python drive.py model-final.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. This file contains the pipeline that I've used to train and validate a deep learning model to autonomously drive the simulation car. Comments are included to further explain my implementation.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I implemented the NVIDIA End-to-End learning model created by Bojarski, et al. published on arxiv.org. Here is a link to the original [paper](https://arxiv.org/pdf/1604.07316.pdf). 

I chose this model for its simplicity and ease of implementation. It was very exciting being able to recreate a formal publication authored by a team of top researchers from one of the foremost companies in deep learning and AI. 

The model consists of 5 convolutional layers that are followed by 3 fully-connected layers, producing a final output value that corresponds to a predicted steering angle. The first layer applies a standard normalization for a given image x of x / 255 - 0.5. This is implemented using a Keras Lambda layer. Input images have a size of 320x160. In addition to normalization, I also crop images to a size of 320x65. I eliminate 75 pixels from the top to eliminate any unnecessary image data such as the sky or mountains and 25 pixels from the bottom of the image to remove the car appearing in the bottom of the frame. 

For the first three convolutional layers, the model employs a 5x5 kernel with a 2x2 stride. For the last two convolutional layers, the model simply employs a 3x3 kernel. The first of the fully-connected layers has an input of 1164 features, the second layer has an input of 100 features, and the third layer has an input of 10 features, resulting in an output of a single value.  

RELU activations were used on all convolutional layers. 

#### 2. Attempts to reduce overfitting in the model
I did not apply any attempts to reduce overfitting in the model. 

Here is a screenshot of my AWS EC2 console displaying training and validation losses. 

In 3 epochs, the model significantly drops in training and validation losses. This did worry me. However, after testing on the track, I was able to verify that the model could stay on the track for multiple laps without crossing over valid lane lines or edges. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 69). I trained the model for 3 epochs, with a train-test split percentage of 80-20. I chose a MSE loss, mainly because I viewed this model as simply being a powerful regression model. 

#### 4. Appropriate training data

I used Udacity's given data set only. I did implement some basic augmentation that I will describe below. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was primarily out of a strong desire to recreate a known model with real-world application and effectiveness. Initially, I had a desire to use a much larger network like VGG16. However, after reading through Nvidia's implementation, I decided to implement the model myself. For my first implementation of the project, I used my LeNet implementation from Project 2. However, I wanted to try out something different (though it didn't necessarily report terrible results).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model (LeNet5) had a low mean squared error on the training set and a relatively low mean squared error on the validation set. I didn't really concern myself too much about overfitting, as this implementation worked quite well for Project 2. However, I decided to try out the Nvidia model instead, and I found it worked really nicely. I didn't have to make any major changes to the model architecture, which was very nice. 

The final step was to run the simulator to see how well the car was driving around track one. The vehicle was able to drive autonomously around the track without leaving the road! 

I let the model drive for multiple laps and recorded the images. Here's a link to the [video](./video-final.mp4): 

#### 2. Final Model Architecture

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I solely relied on the sample driving data provided by Udacity. For pre-processing, I simply converted images from BGR to RGB color space. In addition, I utilized image data from all 3 cameras. With a basic correction factor of 0.2, I appended the associated steering wheel angle with a correction of 0.2. Furthermore, I flipped all images and produced corresponding steering wheel measures (multiply by -1).  

I finally randomly shuffled the data set and put 80% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I chose the split percentages based on widespread advice of researchers and ML practitioners. I trained my model for 3 epochs with an Adam optimizer using MSE as loss.

#### 4. Simulation
Here is the link to my final video [output](./video-final.mp4). No tire leaves the drivable portion of the track surface, and the car does not pop up onto ledges nor rolls over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).
