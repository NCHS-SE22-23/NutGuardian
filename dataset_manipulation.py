import tensorflow as tf
from keras.utils import image_dataset_from_directory

#Splits the dataset for training and validation
def chop_vegetables(path):
    training = image_dataset_from_directory(
        path,
        seed=69420,
        validation_split=0.2,
        subset='training',
        batch_size=128,
        image_size=(256,256)
    )

    validation = image_dataset_from_directory(
        path,
        seed=69420,
        validation_split=0.2,
        subset='validation',
        batch_size=128
    )

    at = tf.data.AUTOTUNE

    training = training.cache().prefetch(buffer_size=at)
    validation = validation.cache().prefetch(buffer_size=at)

    return training, validation