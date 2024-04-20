from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def load_train(path):
    data_generator = ImageDataGenerator(rescale=1. / 255)
    data_generator_flow = data_generator.flow_from_directory(path,
                                                             target_size=(150, 150),
                                                             batch_size=16,
                                                             class_mode='sparse',
                                                             seed=12345,
                                                             subset='training',
                                                             )
    return data_generator_flow


def create_model(input_shape):
    model = Sequential()


    model.add(Conv2D(filters=6,
                     kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(AvgPool2D())
    model.add(Conv2D(filters=16,
                     kernel_size=(3, 3),
                     activation='relu',
                     ))
    model.add(AvgPool2D())
    model.add(Flatten())
    model.add(Dense(units=120,
                    activation='relu'))
    model.add(Dense(units=84,
                    activation='relu'))
    model.add(Dense(units=12,
                    activation='softmax'))

    optimizer = Adam()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    model.summary()

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=3,
                steps_per_epoch=None, validation_steps=None):

    model.fit(train_data,
              validation_data=test_data,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              batch_size=batch_size,
              verbose=2,
              )

    return model
