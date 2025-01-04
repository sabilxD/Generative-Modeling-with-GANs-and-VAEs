
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 100
batch_size = 64
epochs = 50000

# Generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, activation="relu", input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1024, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(28 * 28, activation="tanh"),
        layers.Reshape((28, 28, 1))
    ])
    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

# Loss and optimizers
def get_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy()(tf.ones_like(fake_output), fake_output)

# Loading and preprocessing data
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)

# Build models
generator = build_generator()
discriminator = build_discriminator()
gan = get_gan(generator, discriminator)

# Optimizers
g_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training loop
for epoch in range(epochs):
    for real_images in dataset:
        # Train discriminator
        noise = tf.random.normal([batch_size, latent_dim])
        generated_images = generator(noise)

        with tf.GradientTape() as d_tape:
            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(generated_images, training=True)
            d_loss = discriminator_loss(real_output, fake_output)

        d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

        # Train generator
        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as g_tape:
            generated_images = generator(noise, training=True)
            fake_output = discriminator(generated_images, training=True)
            g_loss = generator_loss(fake_output)

        g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    # Log progress
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, D Loss: {d_loss.numpy()}, G Loss: {g_loss.numpy()}")

    # Save generated images
    if epoch % 1000 == 0:
        noise = tf.random.normal([16, latent_dim])
        generated_images = generator(noise, training=False)
        generated_images = (generated_images + 1) / 2  # Rescale to [0, 1]
        plt.figure(figsize=(4, 4))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(generated_images[i, :, :, 0], cmap="gray")
            plt.axis("off")
        plt.show()
