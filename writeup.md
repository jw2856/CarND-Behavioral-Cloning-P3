# **Behavioral Cloning** 

---

** Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[data-distribution]: ./examples/data-distribution.png "Data distribution"
[loss-graph]: ./examples/loss-graph.png "Loss graph"
[center]: ./examples/center-sample.jpg "Center"
[left]: ./examples/left-sample.jpg "Left"
[right]: ./examples/right-sample.jpg "Right"
[center-cropped]: ./examples/center-cropped-sample.png "Center Cropped"
[left-cropped]: ./examples/left-cropped-sample.png "Left Cropped"
[right-cropped]: ./examples/right-cropped-sample.png "Right Cropped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 containing video of the car driving itself around the track

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the Nvidia architecture, which consists of the following layers:

| Layer             |     Description                   | 
|:---------------------:|:---------------------------------------------:| 
| Convolution 5x5x24    | 2x2 stride, RELU activation |
| Convolution 5x5x36    | 2x2 stride, RELU activation |
| Convolution 5x5x48    | 2x2 stride, RELU activation |
| Convolution 3x3x64    | 1x1 stride, RELU activation |
| Convolution 3x3x64    | 1x1 stride, RELU activation |
| Flatten       |                        |
| Fully connected | 100x1 output, RELU activation |
| Fully connected | 50x1 output, RELU activation |
| Fully connected | 10x1 output, RELU activation |
| Fully connected | 1x1 output |

I preprocessed the images by normalizing and mean-centering using a Keras lambda layer. I also cropped the image using a Cropping2D layer to take 50 pixels from the top and 20 pixels from the bottom.

** Original Images **
|Left|Center|Right|
|:--------:|:------------:|:------------:|
|[![left][left]|[![center][center]|[![right][right]|

** Cropped **
|Left|Center|Right|
|:--------:|:------------:|:------------:|
|[![left][left]|[![center][center]|[![right][right]|

#### 2. Attempts to reduce overfitting in the model

The


The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

I trained my model on the Udacity provided data, which contains 8036 samples, with each sample containing three images, one from the center, left and right cameras on the vehicle.

Graphing the distribution of the data by steering angle, we can see that there are many more samples of low steering angles (steering angles between 0 and 0.1, with many of the angles at 0.0).

![Data distribution][data-distribution]

I wrote a function to delete a percentage of these low-numbered measurements, but the ultimate model did not utilize this function.

To augment the Udacity training set, I used the left and right images as well, using a correction factor of 0.2. I also flipped each of the images and measurements (including the left and right images) to balance the data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I followed the suggestions on the Udacity classes











In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]




-----------------------------------------------------------------------

#### 3. Creation of the Training Set & Training Process

I utilized the Udacity training set to create the final model. I had originally recorded many frames of data using the keyboard and mouse, in either direction around the track, including recovery driving. However, my model performed poorly training on this data, though at that point my network was incomplete. I switched to the Udacity data since it was proven and used it for the rest of the project.

For each training session, 20% of the dataset was withheld for validation.

I used mean-squared error to measure the error, and I used the Adam optimizer for optimization with the default learning rate of 1.0e-3.
