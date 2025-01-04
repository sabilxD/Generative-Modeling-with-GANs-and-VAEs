
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 2
batch_size = 64
epochs = 50

# Encoder
def build_encoder():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Flatten()(inputs)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    encoder = tf.keras.Model(inputs, [z_mean, z_log_var], name="encoder")
    return encoder

# Sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Decoder
def build_decoder():
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation="relu")(latent_inputs)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(28 * 28, activation="sigmoid")(x)
    outputs = layers.Reshape((28, 28, 1))(x)
    decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")
    return decoder

# VAE model
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = Sampling()([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.add_loss(kl_loss)
        return reconstructed

# Data loading and preprocessing
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype("float32") / 255.0

# Build models
encoder = build_encoder()
decoder = build_decoder()
vae = VAE(encoder, decoder)

# Compile and train
vae.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))

# Visualize latent space
def plot_latent_space(encoder, n=30, figsize=15):
    digit_size = 28
    scale = 2.0
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap="Greys_r")
    plt.axis("off")
    plt.show()

plot_latent_space(encoder)
