
import os
import random
import math

import tensorflow as tf
import tf_keras as tfk
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

tfkl = tfk.layers
tfpl = tfp.layers
tfd = tfp.distributions

import warnings

# Ignore specific Matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import denoising_vae.constants as consts
import denoising_vae.utils as utils

def load_model_weights(model, path):
    if(os.path.isfile(path)): 
        model.load_weights(path)
        print('Weights loaded from file')

    else:
        print('Failed to load weights from file')

import time

class TimeHistory(tfk.callbacks.Callback):
    def __init__(self):
        super(TimeHistory, self).__init__()  # Ensure proper initialization of the base class
        self.times = []  # List to store the cumulative time
        self.total_time = 0.0  # Initialize the total time
        self.epoch_start_time = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        self.total_time += epoch_duration
        self.times.append(self.total_time)
        if logs is not None:
            logs['elapsed_time'] = self.total_time

def run_model(model, train_ds, val_ds, epochs, name='Model', plot=True, time=False):

    # Create a callback that saves the model's weights
    # cp_callback = tfk.callbacks.ModelCheckpoint(filepath=model_path,
    #                                                 save_weights_only=True,
    #                                                 verbose=1,
    #                                                 save_best_only=True,
    #                                                 mode='auto')

    if time:
        time_callback = TimeHistory()
        history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[time_callback])
    else:
        history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

    if plot:
        # Plot history
        
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')

        plt.title(f'{name} Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc="upper left")

        plt.show()
    else:
        return history


def val_model(model, ):
    fig_width, fig_height = 5.0, 2.5 * 4
    figure, ax = plt.subplots(4, 2, figsize=(fig_width, fig_height), constrained_layout=True)

    images = utils.load_examples()

    i = 0
    for image in images:
        x = tf.expand_dims(image, axis=0)
        xhat = model(x)
        reconstruction = xhat[0]

        reconstruction = tf.clip_by_value(reconstruction, 0.0, 1.0)

        ax[i, 0].imshow(image)
        ax[i, 0].axis('off')
        ax[i, 1].imshow(reconstruction)
        ax[i, 1].axis('off')

        i += 1

    plt.show()


def gen_image(model, plot=True, min=-3, max=3):
    if plot:
        figure, ax = plt.subplots(figsize=(2.5, 2.5))

        plt.axis('off')
    mean = [0,0]
    cov = [[1,0],[0,1]]
    xi, yi = np.random.multivariate_normal(mean, cov)

    z_sample = np.array([[xi, yi]])
    x_decoded = model.decoder.predict(z_sample, verbose=0)
    image = x_decoded[0].reshape(consts.IMAGE_SIZE, consts.IMAGE_SIZE, 3)

    if plot:
        plt.imshow(image)
        plt.axis('off')

        plt.show()

    return image

def gen_images(model, count=16, min=-3, max=3):
    columns = 4
    rows = (count + 3) // 4  # Wrap after 4 cols

    figure, ax = plt.subplots(figsize=(columns * 2.5, rows * 2.5))

    plt.axis('off')

    for i in range(count):
        xi = random.uniform(min, max)
        yi = random.uniform(min, max)
        
        z_sample = np.array([[xi, yi]])
        x_decoded = model.decoder.predict(z_sample, verbose=0)
        image = x_decoded[0].reshape(consts.IMAGE_SIZE, consts.IMAGE_SIZE, 3)

        plt.subplot(rows, columns, i + 1)
        plt.imshow(image)
        plt.axis('off')

    plt.show()

def plot_latent_space(model, n=30, figsize=15, start=-1, stop=1):
    # display a n*n 2D manifold of digits
 
    figure = np.zeros((consts.IMAGE_SIZE * n, consts.IMAGE_SIZE * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(start, stop, n)
    grid_y = np.linspace(start, stop, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = model.decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(consts.IMAGE_SIZE, consts.IMAGE_SIZE, 3)
            figure[
                i * consts.IMAGE_SIZE : (i + 1) * consts.IMAGE_SIZE,
                j * consts.IMAGE_SIZE : (j + 1) * consts.IMAGE_SIZE,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = consts.IMAGE_SIZE // 2
    end_range = n * consts.IMAGE_SIZE + start_range
    pixel_range = np.arange(start_range, end_range, consts.IMAGE_SIZE)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
    plt.show()

def save_latent_space(model, model_name, n=30, figsize=15, start=-1, stop=1):
    # display a n*n 2D manifold of digits
 
    figure = np.zeros((consts.IMAGE_SIZE * n, consts.IMAGE_SIZE * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(start, stop, n)
    grid_y = np.linspace(start, stop, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = model.decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(consts.IMAGE_SIZE, consts.IMAGE_SIZE, 3)
            figure[
                i * consts.IMAGE_SIZE : (i + 1) * consts.IMAGE_SIZE,
                j * consts.IMAGE_SIZE : (j + 1) * consts.IMAGE_SIZE,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = consts.IMAGE_SIZE // 2
    end_range = n * consts.IMAGE_SIZE + start_range
    pixel_range = np.arange(start_range, end_range, consts.IMAGE_SIZE)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])

    plt.savefig(f'res/latents/{model_name}.png')