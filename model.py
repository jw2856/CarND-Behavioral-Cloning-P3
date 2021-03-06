import csv
import cv2
import numpy as np
import random
import matplotlib

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

matplotlib.use('Agg')
import matplotlib.pyplot as plt

##### Constants ######
SIDE_CAMERA_CORRECTION = 0.2

# Bins for plotting histogram of steering angles
BINS = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
            0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Probability of keeping  data between 0 and 0.1
KEEP_PROB = .3

##### Functions #######

def readDrivingLog(file):
    lines = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for index, line in enumerate(reader):
            if index == 0: # exclude the first title line
                continue
            lines.append(line)
    return lines

def get_img(path):
    filename = path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Save a histogram plot of the distribution of samples
# ----------------------------------------------------
def plotHistogram(lines, file):
    steeringAngles = []

    for index, line in enumerate(lines):
        steeringAngles.append(float(line[3]))

    hist, bins = np.histogram(steeringAngles, BINS)
    print('average samples per bin', len(steeringAngles)/len(BINS))
    print(hist, bins)

    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.savefig(file)
    plt.gcf().clear()

# Balance data by removing a percentage of steering samples that are between
# 0/0 and 0.1
# --------------------------------------------------------------------------
def balanceData(lines):
    newLines = []

    for index, line in enumerate(lines):
        if ((float(line[3]) >= 0 and float(line[3]) < 0.1) and (random.random() > KEEP_PROB)):
            continue

        newLines.append(line)

    return newLines

# Flip image
# ----------
def getFlippedImage(image):
    return cv2.flip(image, 1)

# Flip measurement
# ----------------
def getFlippedMeasurement(measurement):
    return measurement*-1

# Load the images and measurements
# --------------------------------
def loadImagesAndMeasurements(lines):
    images, measurements = [], []
    
    for line in lines:

        # images
        center_img = get_img(line[0])
        left_img = get_img(line[1])
        right_img = get_img(line[2])

        # measurements
        steering_center = float(line[3])
        steering_left = steering_center + SIDE_CAMERA_CORRECTION
        steering_right = steering_center - SIDE_CAMERA_CORRECTION

        images.extend([
            center_img, getFlippedImage(center_img),
            left_img, getFlippedImage(left_img),
            right_img, getFlippedImage(right_img)])

        measurements.extend([
            steering_center, getFlippedMeasurement(steering_center),
            steering_left, getFlippedMeasurement(steering_left),
            steering_right, getFlippedMeasurement(steering_right)])

    return images, measurements

# Preprocessing layers for the images. Consists of normalization and 
# mean-centering, followed by a cropping layer.
# ------------------------------------------------------------------
def preprocess(model):
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    return model

# Keras implementation of the Nvidia model
# ----------------------------------------
def nvidia():
    model = Sequential()
    model = preprocess(model)
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
    model.compile(loss='mse', optimizer='adam')

    return model

# Save the training and validation loss for each epoch
# ----------------------------------------------------
def saveLossGraph(historyObject, file):
    plt.plot(historyObject.history['loss'])
    plt.plot(historyObject.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(file)
    plt.gcf().clear()

##### Main Script #####
# ---------------------

lines = readDrivingLog('./data/driving_log.csv')
plotHistogram(lines, 'data-distribution.png')
# lines = balanceData(lines)
# plotHistogram(lines, 'updated-steering-angles-histogram.png')
images, measurements = loadImagesAndMeasurements(lines)

X_train = np.array(images)
y_train = np.array(measurements)

model = nvidia()
history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2, verbose=1)

saveLossGraph(history, 'loss-graph.png')

# model.save('model.h5')