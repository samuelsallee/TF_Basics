import math
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.optimizer_v1 import Adam
import numpy as np
import random
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution() # allows the older version of the Adam optimizer to work


x_train = [random.random() * math.pi * 2 for _ in range(100000)]  # generates numbers between 0 and 2*pi (this step takes some time)
y_train = [math.sin(x) for x in x_train]                          # correct outputs for the above inputs

graph_x = [random.random() * math.pi * 2 for _ in range(1000)]  # just for making a sine wave in pyplot
graph_y = [math.sin(x) for x in graph_x]
plt.plot(graph_x, graph_y, 'o')  # draws the sin wave on a pyplot, these are the blue dots


# builds a fully connected NN. Uses Rectified linear activation, the Adam optimizer, and Mean Squared Error for the loss
def BuildModel():
    model = Sequential([
        Input(1),
        Dense(128, activation='relu'),
        Dense(1, dtype=np.float32)
                        ])
    model.compile(loss='mse', metrics=['accuracy'], optimizer=Adam(lr=.001))
    return model


if __name__ == "__main__":
    model = BuildModel()
    model.fit(x_train, y_train, verbose=2, epochs=50, batch_size=64)  # training cycle of the NN
    
    i_test = [random.random() * math.pi * 2 for _ in range(100)]
    pred = model.predict(i_test)

    pred_y = []
    for x in pred:
        pred_y.append(x)
    plt.plot(i_test, pred_y, 'o')  # plots the predicted value next to the correct sine wave for comparison
    plt.show()

