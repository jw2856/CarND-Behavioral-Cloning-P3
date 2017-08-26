import csv
import cv2
import numpy as np

lines = []

with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []

def get_img(path):
  filename = path.split('/')[-1]
  current_path = './data/IMG/' + filename
  image = cv2.imread(current_path)
  return image

for index, line in enumerate(lines):
  if index == 0:
    continue

  center_img = get_img(line[0])
  left_img = get_img(line[1])
  right_img = get_img(line[2])

  images.extend(center_img, left_img, right_img)

  correction = 0.2
  steering_center = float(line[3])
  steering_left = steering_center + correction
  steering_right = steering_center - correction
  
  measurements.append(steering_center, steering_left, steering_right)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image, 1))
  augmented_measurements.append(measurement*-1)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')