import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
            kl_loss = - 0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class AE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            total_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
        }


class IWAE(keras.Model):
    def __init__(self, encoder, decoder, k, **kwargs):
        super(IWAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.k = k
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            for i in range(0, self.k):
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
                post_kld = tf.keras.backend.sum(
                    -0.5 * (z_log_var + tf.keras.backend.square(z - z_mean) / tf.keras.backend.exp(z_log_var)), axis=-1)
                prior_kld = tf.keras.backend.sum(-0.5 * tf.keras.backend.square(z), axis=-1)
                kl_loss = post_kld - prior_kld
                if i == 0:
                    elbo = tf.keras.backend.expand_dims(kl_loss + reconstruction_loss)
                else:
                    elbo = tf.keras.backend.concatenate([elbo, tf.keras.backend.expand_dims(kl_loss + reconstruction_loss)], axis=-1)

            weights = tf.stop_gradient(tf.keras.backend.softmax(- elbo, axis=-1))
            total_loss = tf.reduce_mean(tf.keras.backend.sum(weights * elbo, axis=-1))

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(tf.reduce_mean(reconstruction_loss))
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_loss))
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def get_encoder_decoder(dataset, model_type, posterior, latent_dim):
    if dataset == 'mnist' or dataset == 'fashion':
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        x = layers.Flatten()(encoder_inputs)
        x = layers.Dense(200, activation="tanh")(x)
        x = layers.Dense(200, activation="tanh")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)

        if model_type != 'AE':
            if posterior == 'dgp':
                z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
            elif posterior == 'igp':
                z_log_var = layers.Dense(1, name="z_log_var")(x)
                z_log_var = tf.keras.backend.repeat_elements(z_log_var, z_mean.shape[1], axis=1)
            else:
                raise ValueError('Should use \'igp\' or \'dgp\' for posterior!')

            z = Sampling()([z_mean, z_log_var])
            encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        else:
            encoder = keras.Model(encoder_inputs, z_mean, name='encoder')
        encoder.summary()

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(200, activation="tanh")(latent_inputs)
        x = layers.Dense(200, activation="tanh")(x)
        x = layers.Dense(28 * 28 * 1, activation='sigmoid')(x)
        decoder_outputs = layers.Reshape((28, 28, 1))(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
    else:
        encoder_inputs = keras.Input(shape=(64, 64, 1))
        x = layers.Conv2D(32, 4, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(32, 4, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(64, 4, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(64, 4, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)

        if model_type != 'AE':
            if posterior == 'dgp':
                z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
            elif posterior == 'igp':
                z_log_var = layers.Dense(1, name="z_log_var")(x)
                z_log_var = tf.keras.backend.repeat_elements(z_log_var, z_mean.shape[1], axis=1)
            else:
                raise ValueError('Should use \'igp\' or \'dgp\' for posterior!')

            z = Sampling()([z_mean, z_log_var])
            encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        else:
            encoder = keras.Model(encoder_inputs, z_mean, name='encoder')
        encoder.summary()

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(256, activation="relu")(latent_inputs)
        x = layers.Dense(4 * 4 * 64, activation='relu')(x)
        x = layers.Reshape((4, 4, 64))(x)
        x = layers.Conv2DTranspose(64, 4, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(64, 4, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 4, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 4, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 4, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

    return encoder, decoder


if __name__ == '__main__':
    latent_dim = 32
    epochs = 50
    batch_size = 128

    n = 10
    figsize = 15

    model_type = 'VAE'
    k = 50
    posterior = 'igp'
    dataset = 'mnist'

    if dataset == 'mnist':
        digit_size = 28
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    elif dataset == 'fashion':
        digit_size = 28
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    else:
        digit_size = 64
        exit()

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255

    for seed in range(1):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        encoder, decoder = get_encoder_decoder(dataset, model_type, posterior, latent_dim)

        if model_type == 'AE':
            model = AE(encoder, decoder)
        elif model_type == 'VAE':
            model = VAE(encoder, decoder)
        else:
            model = IWAE(encoder, decoder, k)

        model.compile(optimizer=keras.optimizers.Adam())
        model.fit(x_train, epochs=epochs, batch_size=batch_size)

        if model_type == 'AE':
            z = model.encoder.predict(x_test)
            reconstruction = model.decoder.predict(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(x_test, reconstruction), axis=(1, 2)))
        else:
            z_mean, z_log_var, z = model.encoder.predict(x_test)
            reconstruction = model.decoder.predict(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(x_test, reconstruction), axis=(1, 2)))
            kl_loss = - 0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        if model_type != 'AE':
            z_train = model.encoder.predict(x_train)[0]
            z_test = model.encoder.predict(x_test)[0]
        else:
            z_train = model.encoder.predict(x_train)
            z_test = model.encoder.predict(x_test)

        z_all = tf.keras.backend.concatenate([z_train, z_test], axis=0).numpy()
        cov = np.cov(z_all, rowvar=False)
        s = []
        std = np.std(z_all, axis=1)
        for i in range(0, cov.shape[0]):
            if cov[i][i] > 0.01:
                s.append(i + 1)

        if dataset == 'mnist' or dataset == 'fashion':
            cls_model = keras.Sequential(
                    [
                        keras.Input(shape=(latent_dim,)),
                        layers.Dense(num_classes, activation="softmax"),
                    ]
                )
            cls_model.summary()

            cls_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
            cls_model.fit(z_train, y_train, batch_size=batch_size, epochs=30, validation_split=0.1)

            score = cls_model.evaluate(z_test, y_test, verbose=0)
            print(f'Test accuracy: {score[1]:.4f}')
        else:
            score = [0, 0]

        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space

        for i in range(n):
            for j in range(n):
                z_sample = np.random.normal(size=(1, latent_dim))
                x_decoded = model.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(figsize, figsize))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 20
        if model_type == 'VAE':
            prefix = f'model/{dataset}/vae_{posterior}_{seed}_{total_loss:.2f}_{reconstruction_loss:.2f}'\
                     f'_{kl_loss:.2f}_{len(s)}_{score[1]:.4f}'
        elif model_type == 'IWAE':
            prefix = f'model/{dataset}/iwae_{k}_{posterior}_{seed}_{total_loss:.2f}_{reconstruction_loss:.2f}'\
                     f'_{kl_loss:.2f}_{len(s)}_{score[1]:.4f}'
        else:
            prefix = f'model/{dataset}/ae_{seed}_{reconstruction_loss:.2f}_{len(s)}_{score[1]:.4f}'
        plt.imshow(figure, cmap="Greys")
        plt.axis('off')
        plt.savefig(prefix + '_white.pdf')
        plt.clf()
        plt.imshow(figure, cmap="Greys_r")
        plt.axis('off')
        plt.savefig(prefix + '_black.pdf')
        plt.close()
