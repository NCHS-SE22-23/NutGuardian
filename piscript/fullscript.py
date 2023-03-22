# Neil Ganguly, Harry Yu, Kynan McCabe-Wild, Collin O'Connor
# Nut Guardian
# This is the python file that will go into the raspberry pi and run the code that we need to ID squirrels vs birds
# 2/6/2023

# Imports
import numpy as np
import time
import subprocess
import shutil
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import load_img, img_to_array
from pathlib import Path
import cv2
from gpiozero import MotionSensor
import pygame

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

# Setting up the GPIO board so we can use the pin board
pir = MotionSensor(2) # the int is the power pin for the motionsensor
vid = cv2.VideoCapture(0)

# Setting up the audio system assuming we have already set the default device
pygame.mixer.init()
sound = pygame.mixer.Sound("whitenoise.mp3")

# Once we detect motion, take pictures and send them to the neural network
while True:
    pir.wait_for_motion(timeout=None)

    # Recording the 30 pictures 
    ret, frame = vid.read()

    # Creates the path for frames
    path = Path('frames').absolute()

    # Makes the directory and folder
    path.mkdir()

    # Converts video to 30 frames
    subprocess.call('ffmpeg -i video.h264 -vf fps=10 frames/frame%03d.jpg')

    # Adds all the frames to the frames folder
    frames = Path.glob('frames/*.*')

    # Comparing number of preds to squirrels vs birds
    squirrel_count = 0

    for count, img_path in enumerate(list(frames)):
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

    # Removes the folder afterwards
    shutil.rmtree(path)

    # If determined to be a squirrel, play the speaker noise
    if (squirrel_count > 15):
        playing = sound.play()
        while playing.get_busy():
            pygame.time.delay(10000)
            
    # Wait for the next instance of no motion to return to the start of the loop
    pir.wait_for_no_motion(timeout=None)