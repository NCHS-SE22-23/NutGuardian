from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPooling2D, Conv2D, Rescaling, Activation, Dropout

def munch():
    model = Sequential()

    model.add(Rescaling(1.0/255))

    model.add(Conv2D(32, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(64, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(1, activation='sigmoid'))

    return model