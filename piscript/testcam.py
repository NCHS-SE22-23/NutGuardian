from picamera import PiCamera
from gpiozero import MotionSensor
from time import sleep
import pygame
from pygame.locals import *

print("started")

# Initialize the camera
camera = PiCamera()

# Settings for the camera
camera.resolution = (128, 128)
camera.framerate = 60
camera.brightness = 50
camera.saturation = 50

print("got here")

# Setting up the motion sensor
pir = MotionSensor(15)

print("got here 2")

pygame.init()
pygame.mixer.music.load('/home/nutguardian/Desktop/nut-guardian/NutGuardian/piscript/whitenoise.mp3')

while True:

    pir.wait_for_motion()

    print("motion detected")

    for i in range(15):
        camera.capture('/home/nutguardian/Desktop/nut-guardian/NutGuardian/piscript/pic{}.jpg'.format(i))
    
    
    sleep(7)#wait for model to detect squirrel


    #pygame.mixer.music.play()
    print("played audio")
    sleep(7)
    #pygame.mixer.music.stop()


    pir.wait_for_no_motion