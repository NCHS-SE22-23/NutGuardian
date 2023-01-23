import time
import matplotlib.pyplot as plt
from model import create_model
from data_creator import download_flickr_data

start = time.time()

#Downloading the data
#download_flicker_data()

#Creating and training the model
model_history = create_model(50)

#Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(model_history.history['accuracy'], label='Training Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(model_history.history['loss'], label='Training Loss')
plt.plot(model_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

end = time.time()

print("Total time elapsed:", end - start)