import time
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.callbacks import ModelCheckpoint
from model import create_model
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
model = create_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

checkpoint_dir = Path.cwd() / f'trained models/mod1.ckpt'
callback = ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True,
    verbose=1
)

history = model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=[callback])

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