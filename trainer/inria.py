'''A modification of the mnist_mlp.py example on the keras github repo.

This file is better suited to run on Cloud ML Engine's servers. It saves the
model for later use in predictions, uses pickled data from a relative data
source to avoid re-downloading the data every time, and handles some common
ML Engine parameters.
'''

from __future__ import print_function

import argparse
import os
import pickle  # for handling the new data source
import sys
from datetime import datetime  # for filename conventions

import h5py  # for saving the model
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D, BatchNormalization)
from keras.models import Sequential
from keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.lib.io import file_io  # for better file I/O


batch_size = 32
epochs = 15
img_width, img_height = 256, 256
steps_per_epoch = 1800 // batch_size
validation_steps = 740 // batch_size
# Create a function to allow for different training data and other options


def train_model(train_file='inzynierka',
                job_dir='./job_dir',
                cache=False, **args):
    os.system('gsutil cp -r gs://aerfio-bucket/data/inzynierka .')

    # set the logging path for ML Engine logging to Storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))
    train_data_dir = train_file+'/train'
    validation_data_dir = train_file+'/validation'

   # Reading in the pickle file. Pickle works differently with Python 2 vs 3
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('selu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('selu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('selu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('selu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    ear = EarlyStopping(monitor='acc', min_delta=1, patience=4,)
    ron = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=4,)
    model.fit_generator(
        train_generator,
        workers=4,
        epochs=epochs,
        callbacks=[ear, ron],
        validation_data=validation_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
    score = model.evaluate_generator(
        validation_generator, verbose=1, steps=len(validation_generator),)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('model.h5')

    # Save model.h5 on to google storage
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':

    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-file',
        help='Cloud Storage bucket or local path to training data')
    parser.add_argument(
        '--job-dir',
        help='Cloud storage bucket to export the model and store temp files')
    parser.add_argument(
        '--cache',
        help='use cached pics')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
