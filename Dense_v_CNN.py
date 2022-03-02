import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.optimizer_v1 import Adam
from keras.datasets import mnist
import numpy as np

tf.compat.v1.disable_eager_execution()  # allows the older version of the Adam optimizer to work


(x_train, y_train), (x_test, y_test) = mnist.load_data()  # retrieves MNIST dataset

# Normalizing the inputs
x_test = x_test / 255
x_train = x_train / 255

# Building Fully connected NN
def FullyConnected():
    model = Sequential([
        Flatten(),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(10, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr=.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Building Convolutional NN
def CNN():
    model = Sequential([
        Input((28,28,1)),
        Conv2D(32, kernel_size=(3,3), activation='relu'),
        Conv2D(16, kernel_size=(3,3), activation='relu'),
        Flatten(),
        Dense(10, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr=.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':

    # training fully connected model
    FC_model = FullyConnected()
    FC_model.fit(x_train, y_train, epochs=4, verbose=2, batch_size=64)

    # training convolutional model
    CNN_model = CNN()
    CNN_model.fit(np.expand_dims(x_train, axis=-1), y_train, epochs=4, verbose=2, batch_size=64)

    # testing the models
    print(FC_model.evaluate(x_test, y_test, verbose=2))
    print(CNN_model.evaluate(np.expand_dims(x_test, axis=-1), y_test, verbose=2))

