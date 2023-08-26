from keras.models import Sequential
from keras.layers import Flatten, Rescaling, Dense, MaxPooling2D, Conv2D, Dropout, InputLayer
from keras.callbacks import EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt

#Creates a CNN model
def barbecue():
    model = Sequential() 

    model.add(InputLayer(input_shape=(256, 256, 3)))
    model.add(Rescaling(1.0/255))

    model.add(Conv2D(32, 3, activation='relu'))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model

#Compiles and trains the model
def munch(cnn, training, validation):
    cnn.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    earlystopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    return cnn.fit(training, epochs=50, validation_data=validation, callbacks=[earlystopping])

#Plots the accuracies and losses
def taste_test_results(output):
    plt.figure(figsize=(10, 5))
    plt.title("Taste Test Results")

    plt.subplot(1, 2, 1)
    plt.plot(output.history['accuracy'], label='Training Accuracy')
    plt.plot(output.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Dish Number (Epoch)')
    plt.ylabel('Gordon Ramsay Rating (Accuracy)')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(output.history['loss'], label='Training Loss')
    plt.plot(output.history['val_loss'], label='Validation Loss')
    plt.xlabel('Dish Number (Epoch)')
    plt.ylabel('Gordon Ramsay Disliking (Loss)')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    plt.show()

#Saves the weights of the model
def learn_recipe(cnn):
    cnn.save_weights(Path.cwd() / 'trained models/model2.h5')