import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(32, activation="relu")(input_layer)
    x = Dense(16, activation="relu")(x)
    encoded = Dense(8, activation="relu")(x)

    x = Dense(16, activation="relu")(encoded)
    x = Dense(32, activation="relu")(x)
    decoded = Dense(input_dim, activation="linear")(x)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder
