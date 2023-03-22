import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import load_img, img_to_array
from pathlib import Path

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

model.load_weights('model.h5')

plt.figure(figsize=(8.5, 8.5))

for count, img_path in enumerate(Path('/Users/neilganguly/Documents/School-Github/NutGuardian/testing the model/sqrls').iterdir()):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_batch = img_batch / 255.0

    prediction_array = model.predict(img_batch, verbose=0)
    
    prediction = np.argmax(prediction_array[0])

    title = 'bird'

    if prediction == 1:
        title = 'squirrel'

    plt.subplot(4, 5, count + 1)
    plt.imshow(img)
    plt.title('Actual: ' + img_path.stem + '\nPred: ' + title + '\n' + '%.3f'%prediction_array[0][prediction])

plt.tight_layout()
plt.show()