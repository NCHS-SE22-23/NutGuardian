import time
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.callbacks import EarlyStopping
from model import munch
from data_creator import download_flickr_data


#USE TF LITE FOR RASPBERRY PI

start = time.time()

#Downloading the data
#download_flicker_data()

#Splitting the datasets
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

#Creating and training the model
model = munch()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

earlystopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[earlystopping])

model.save('trained models/model1')

#Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

end = time.time()

print("Total time elapsed:", end - start)