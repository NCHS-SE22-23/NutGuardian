import tensorflow as tf
from pathlib import Path
from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPooling2D, Conv2D, Rescaling

#USE TF LITE FOR RASPBERRY PI

def create_model(epochs):
    data = Path('images')

    train_ds = image_dataset_from_directory(
        data,
        seed=69420,
        validation_split=0.2,
        subset='training'
    )

    val_ds = image_dataset_from_directory(
        data,
        seed=69420,
        validation_split=0.2,
        subset='validation'
    )

    at = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=at)
    val_ds = val_ds.cache().prefetch(buffer_size=at)

    model = Sequential()

    model.add(Rescaling(1.0/255))

    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model.fit(train_ds, epochs=epochs, validation_data=val_ds)