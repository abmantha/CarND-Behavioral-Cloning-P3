**Behavioral Cloning Project** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./output_images/visualizing-loss.png 
[image2]: ./output_images/nvidia-model.png
[image3]: ./output_images/all-of-my-attempts.png
[image4]: ./output_images/all-of-my-attempts-2.png
[image5]: ./output_images/all-of-my-attempts-3.png
[image6]: ./output_images/training_data_center_1.jpg
[image7]: ./output_images/training_data_center_2.jpg
[image8]: ./output_images/training_data_center_3.jpg
[image9]: ./output_images/training_data_center_4.jpg
[image10]: ./output_images/actual_data_1.jpg
[image11]: ./output_images/actual_data_2.jpg
[image12]: ./output_images/actual_data_3.jpg

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
I did not apply any attempts to reduce overfitting in this model. In previous versions of this model, I used Dropout layers, additional max pool layers, and ELU units, as per the recommendation of the well-known model by [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py). However, I found that though these additions were leading to smaller and smaller training and validation losses, they were certainly leading to incorrect driving behavior, which is what matters. 

Here is a screenshot of my AWS EC2 console displaying training and validation losses. 

![alt text][image1]

In 3 epochs, the model significantly drops in training and validation losses. This did worry me. However, after testing on the track, I was able to verify that the model could stay on the track for multiple laps without crossing over valid lane lines or edges. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 69). I trained the model for 3 epochs, with a train-test split percentage of 80-20. I chose a MSE loss, mainly because I viewed this model as simply being a powerful regression model. 

#### 4. Appropriate training data

For this model, I used Udacity's given data set only. Here's a link to a video of the collected sample [data](./training_data_video.mp4) I did implement some basic augmentation that I will describe below. 

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was primarily out of a strong desire to recreate a known model with real-world application and effectiveness. Initially, I had a desire to use a much larger network like VGG16. However, after reading through Nvidia's implementation, I decided to implement the model myself. For my first implementation of the project, I used my LeNet implementation from Project 2. However, I wanted to try out something different (though it didn't necessarily report terrible results).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model (LeNet5) had a low mean squared error on the training set and a relatively low mean squared error on the validation set. I didn't really concern myself too much about overfitting, as this implementation worked quite well for Project 2. However, I decided to try out the Nvidia model instead, and I found it worked really nicely. I didn't have to make any major changes to the model architecture, which was very nice. 

The final step was to run the simulator to see how well the car was driving around track one. The vehicle was able to drive autonomously around the track without leaving the road! 

I let the model drive for multiple laps and recorded the images. Here's a link to the [video](./video-final.mp4): 

#### 2. Final Model Architecture

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image2]

#### 3. Creation of the Training Set & Training Process
For this model, I solely relied on the sample driving data provided by Udacity. For pre-processing, I simply converted images from BGR to RGB color space. In addition, I utilized image data from all 3 cameras. With a basic correction factor of 0.2, I appended the associated steering wheel angle with a correction of 0.2. 

I finally randomly shuffled the data set and put 80% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I chose the split percentages based on widespread advice of researchers and ML practitioners. I trained my model for 3 epochs with an Adam optimizer using MSE as loss.

#### 4. Simulation
Here is the link to my final video [output](./video-final.mp4). No tire leaves the drivable portion of the track surface, and the car does not pop up onto ledges nor rolls over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle). Here are actual image frames that the car sees:

![alt text][image10]
![alt text][image11]
![alt text][image12]

### Discussion
This was hands-down the weirdest project I have worked on. In summary, I have gained a deep appreciation for how powerful deep learning can be. In total, I think I spent over 25 hours working on this project. I implemented some fairly basic architectures, but the main focus of this assignment was data. In some of my previous iterations of this project, I applied things like horizontal shifts, brightness augmentation and linear transformations. While I found that my training and validation accuracies were becoming lower and lower, I found that the car's driving behavior was simply just wrong. Numerous times, the car would drive well on straight segments, on some turns it would execute them perfectly, and then randomly, out of no-where, the car would go right off the road. Here are some snapshots of some of previous attempts over the last few weeks: 

![alt text][image3]
![alt text][image4]
![alt text][image5]

What baffled me was how the car was able to do something so unexpected, given that it had never even been in that situation during data collection. This honestly frightens me. For something so powerful, many deep learning models are truly black boxes. I still can't wrap my head around the fact the above model works as well as it does. I was not expecting it to outperform my previous approaches that incorporated techniques reported by other students who designed some very powerful architectures and approaches. 

But at the same time, I think that things like simulation, and even standard tools like feature engineering, are particularly important for the self-driving car industry. Without these techniques and mechanisms, I don't think SDC companies and researchers would have the success that they currently do. 

The above implementation was approximately my 14th attempt to completing this project. Oddly enough, I'm quite surprised that it worked so easily and so seamlessly on track one. 

In retrospect, I might have attempted to build a simpler architecture. There's been chatter of models consisting of minimal dense layers that have produced outstanding results. Simple architecture and minimal training cost with high efficiency is a top priority for the field. Also, I'd like to have incorporated more data from the second track. As it stands, my model will fail pretty quickly and fails to generalize well. Generalizing this model is definitely something I'd like to tackle when I re-visit this project in the future. 
