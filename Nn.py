import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load the data
data = np.loadtxt("optdigits_train.dat")
data_test = np.loadtxt("optdigits_test.dat")
data_trial = np.loadtxt("optdigits_trial.dat")

# Prepare the data
m_train = data.shape[0]
X_train = data[:, :-1].reshape(m_train, 32, 32, 1)
y_train = to_categorical(data[:, -1].reshape(m_train, 1))

m_test = data_test.shape[0]
X_test = data_test[:, :-1].reshape(m_test, 32, 32, 1)
y_test = to_categorical(data_test[:, -1].reshape(m_test, 1))

m_trial = data_trial.shape[0]
X_trial = data_trial[:, :-1].reshape(m_trial, 32, 32, 1)
y_trial = to_categorical(data_trial[:, -1].reshape(m_trial, 1))

# Define the model
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 1)))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32
)

model.save("digit_recognition_model.h5", include_optimizer=False, save_format="h5")
