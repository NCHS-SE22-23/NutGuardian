# Neil Ganguly, Harry Yu, Kynan McCabe-Wild, Collin O'Connor
# Nut Guardian
# This is the python file that will go into the raspberry pi and run the code that we need to ID squirrels vs birds
# 2/6/2023

# Imports
import numpy as np
import time
import subprocess
import shutil
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import load_img, img_to_array
from pathlib import Path
from gpiozero import MotionSensor
import pygame
from picamera import PiCamera
from gpiozero import MotionSensor
from time import sleep
import pygame
from pygame.locals import *

# Instantiate the model to get it running
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Loading the relevant weights so the model will work
model.load_weights('../model.h5')

# Setting up the peripherals
camera = PiCamera()

camera.resolution = (128, 128)
camera.framerate = 60
camera.brightness = 50
camera.saturation = 50

pir = MotionSensor(15)

pygame.init()
pygame.mixer.music.load('/home/nutguardian/Desktop/nut-guardian/NutGuardian/piscript/whitenoise.mp3')

# Once we detect motion, take pictures and send them to the neural network
while True:
    pir.wait_for_motion(timeout=None)

    i = 0

    # Recording the 15 pictures 
    for i in range(15):
        camera.capture('/home/nutguardian/Desktop/nut-guardian/NutGuardian/piscript/pic{}.jpg'.format(i))

    # Comparing number of preds to squirrels vs birds
    squirrel_count = 0

    i = 0

    # Determine squirrel
    for i in range(15):
        img_path = '/home/nutguardian/Desktop/nut-guardian/NutGuardian/piscript/pic' + i + '.jpg'
        img = mpimg.imread(img_path)
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_batch = img_batch / 255.0

        # Model predicts the picture
        prediction_array = model.predict(img_batch, verbose=0)
        
        # Highest number of dense neurons is the output
        prediction = np.argmax(prediction_array[0])

        if prediction == 1:
            squirrel_count += 1


    # If determined to be a squirrel, play the speaker noise
    if (squirrel_count > 7):
        #pygame.mixer.music.play()
        sleep(7)
        #pygame.mixer.music.stop()
        print("SQRIILWEL")
    else:
        print("BIRB")
            
    # Wait for the next instance of no motion to return to the start of the loop
    pir.wait_for_no_motion(timeout=None)