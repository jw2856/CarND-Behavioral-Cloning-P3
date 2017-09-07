# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[data-distribution]: ./examples/data-distribution.png "Data distribution"
[balanced-data]: ./examples/updated-steering-angles-histogram.png "Balanced data distribution"
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

#### 1. Model Architecture

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

**Original Images**

| Left | Center | Right |
|:--------:|:------------:|:------------:|
| ![left][left] | ![center][center] | ![right][right] |

**Cropped Images**

| Left | Center | Right |
|:--------:|:------------:|:------------:|
| ![left cropped][left-cropped] | ![center cropped][center-cropped] | ![right cropped][right-cropped] |

#### 2. Training Data

I trained my model on the Udacity provided data, which contains 8036 samples, with each sample containing three images, one from the center, left and right cameras on the vehicle.

Graphing the distribution of the data by steering angle, we can see that there are many more samples of low steering angles (steering angles between 0 and 0.1, with many of the angles at 0.0).

![Data distribution][data-distribution]

To augment the Udacity training set, I used the left and right images as well, using a correction factor of 0.2. I also flipped each of the images and measurements (including the left and right images) to balance the data.

### Model Architecture and Training Strategy

I initially gathered my own data and used it on a variety of iterations of both the LeNet model and the Nvidia model. The LeNet model performed very poorly and so I continued with the Nvidia model and tweaking parameters, including attempting to balance the data, adding dropout layers, and adjusting the number of epochs.

I had originally recorded many frames of data using the keyboard and mouse, in either direction around the track, including recovery driving. However, my model performed poorly training on this data, though at that point my network was incomplete. I switched to the Udacity data since it was proven and used it for the rest of the project.

I wrote a function to delete a percentage of these low-numbered measurements. Depending on the keep probability constant, a balanced dataset might look like this:

![Balanced data][balanced-data-distribution]

The final model did not utilize this function, as it was unnecessary.

Ultimately, I found a successful model by utilizing the Udacity-provided data, and reverting to the straightforward approach recommended by the Nvidia model. The data augmentation steps that I took on the Udacity data resulted in a training dataset of 38,572 samples. The model did not need very many epochs to train, as the training loss was very low even after one epoch. The validation loss would not predictably decrease with more epochs.

For each training session, 20% of the dataset was withheld for validation.

I used mean-squared error to measure the error, and I used the Adam optimizer for optimization with the default learning rate of 1.0e-3.

The loss graph for the model is below:

![Loss graph][loss-graph]

### Improvements

To improve the model, there are certain approaches that I can take in the future:
- Use a better method to balance the data, rather than the naive approach in the unused function I wrote.
- Use techniques to generate random variations on the training data, such as rotating the images, adding shadows, etc.
- Gather and train the model on the second track to see how the model works in a more complex environment.
