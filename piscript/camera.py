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
from picamera import PiCamera

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

model.load_weights('../model.h5')

for i in range(5):
    camera = PiCamera()
    camera.framerate = 10

    camera.start_recording('video.h264')
    time.sleep(3)
    camera.stop_recording()

    path = Path('frames').absolute()

    path.mkdir()

    subprocess.call('ffmpeg -i video.h264 -vf fps=10 frames/frame%03d.jpg')

    frames = Path.glob('frames/*.*')

    for count, img_path in enumerate(list(frames)):
        img = mpimg.imread(img_path)
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_batch = img_batch / 255.0

        prediction_array = model.predict(img_batch, verbose=0)

        prediction = np.argmax(prediction_array[0])

        title = 'bird'

        if prediction == 1:
            title = 'squirrel'

        plt.subplot(6, 5, count + 1)
        plt.imshow(img)
        plt.title('Pred: ' + title + '\n' + '%.3f'%prediction_array[0][prediction])

    plt.tight_layout()
    plt.show()

    shutil.rmtree(path)

    time.sleep(10)