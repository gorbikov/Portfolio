from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow import random


def load_train(path):
    data_generator = ImageDataGenerator(rescale=1. / 255,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        rotation_range=90,
                                        zoom_range=0.2
                                        )
    data_generator_flow = data_generator.flow_from_directory(path,
                                                             target_size=(150, 150),
                                                             batch_size=16,
                                                             class_mode='sparse',
                                                             seed=12345,
                                                             subset='training',
                                                             )
    return data_generator_flow


def create_model(input_shape):

    backbone = ResNet50(input_shape=input_shape,
                        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(12, activation='softmax'))

    initial_learning_rate = 5 * (10**-4)
    print(initial_learning_rate)
    decay_rate = 0.5
    print(decay_rate)


    lr_schedule = ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=1463,
        decay_rate=decay_rate)

    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    model.summary()

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=3,
                steps_per_epoch=None, validation_steps=None):

    random.set_seed(12345)

    model.fit(train_data,
              validation_data=test_data,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              batch_size=batch_size,
              verbose=2,
              )

    return model
