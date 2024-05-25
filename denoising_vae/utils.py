
import os
import random
import math

import tensorflow as tf
import tf_keras as tfk
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
# from scipy.ndimage import gaussian_filter


tfkl = tfk.layers
tfpl = tfp.layers
tfd = tfp.distributions

import denoising_vae.constants as consts

def apply_noise(image, level):
    if level == 0:
        return image
    
    noising_schedule_tensor = tf.constant(consts.NOISING_SCHEDULE, dtype=tf.float32)
    
    alpha_t_sequence = 1 - noising_schedule_tensor[:level]
    alpha_t = tf.math.cumprod(alpha_t_sequence)[-1]

    mean = tf.sqrt(alpha_t) * image
    sigma = tf.sqrt(1 - alpha_t)

    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=sigma, dtype=image.dtype)
    
    noised_image = mean + noise
    return noised_image

def compare_images(imageA, imageB):
    psnr = PSNR(imageA, imageB)
    ssim = ssim_index(imageA, imageB)
    return psnr, ssim

def PSNR(imageA, imageB):
    mse = np.mean((imageA - imageB) ** 2)
    # If they were to be exactly the same
    if mse == 0:
        return 100
    # Max value 
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def ssim_index(imageA, imageB):
    # Convert images to grayscale based on luminance (Y)
    y = 0.299 * imageA[:, :, 0] + 0.587 * imageA[:, :, 1] + 0.114 * imageA[:, :, 2]
    x = 0.299 * imageB[:, :, 0] + 0.587 * imageB[:, :, 1] + 0.114 * imageB[:, :, 2]

    # Constants for SSIM index formula
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Gaussian Kernel
    window_size = 11
    sigma = 1.5  # Standard deviation for Gaussian kernel

    # Gaussian window/weights
    window = np.outer(np.hanning(window_size), np.hanning(window_size))
    window /= window.sum()

    # Local means using Gaussian weights
    mu_x = 0 #gaussian_filter(x, sigma)
    mu_y = 0 #gaussian_filter(y, sigma)

    # Local variance and covariance
    sigma_x2 = 0 #gaussian_filter(x**2, sigma) - mu_x**2
    sigma_y2 = 0 #gaussian_filter(y**2, sigma) - mu_y**2
    sigma_xy = 0 #gaussian_filter(x * y, sigma) - mu_x * mu_y

    # Calculate SSIM index
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den

    return np.mean(ssim_map)

def load_noise_examples():

    print('Loading examples...')

    
    folder_path = f'res/images/{consts.IMAGE_SIZE}'
    
    files = tf.data.Dataset.list_files(str(f'{folder_path}/*'), shuffle=False).take(consts.NOISE_LEVELS)

    input_images = []
    output_images = []
    levels = []

    level = 0

    for file in files:
        if level % 16 == 0:
            image = tf.io.read_file(file)
            image = decode_img(image)
            
            noised_image = apply_noise(image, level)
            input_images.append(image)
        
            output_images.append(noised_image)
        
            levels.append(level)

        level += 1
    
    return input_images, output_images, levels

def load_noised_test_images():
    folder_path = f'res/images/{consts.IMAGE_SIZE}t'
    
    files = tf.data.Dataset.list_files(str(f'{folder_path}/*'), shuffle=False)

    input_images = []
    noised_images = []

    levels = []

    print('Loading images...')

    for file in files:
       
        image = tf.io.read_file(file)
        image = decode_img(image)

        level = np.random.random_integers(1, consts.NOISE_LEVELS)
        noised_image = apply_noise(image, level)

        input_images.append(image)
        noised_images.append(noised_image)
        levels.append(level)
      
    
    return input_images, noised_images, levels


def validate_denoising(model):
    clean, noisy, levels = load_noised_test_images()
    results = []

    print('Validating denoising...')


    for i in range(len(clean)): 
        if(i % 50 == 0):
            print(f'Scoring image {i}...')
        #x = tf.expand_dims(noisy[i], axis=0)
        #xhat = model(x)
        #reconstruction = xhat[0]
        #reconstruction = tf.clip_by_value(reconstruction, 0.0, 1.0)
        reconstruction = noisy[i]

        #scale to 0-255
        clean[i] = clean[i] * 255
        reconstruction = reconstruction * 255

        psnr, ssim = compare_images(clean[i], reconstruction)
        results.append((psnr, ssim, levels[i], i))
    
    return results


def plot_noise_levels():
    
    image, level_images, levels = load_noise_examples()
    col = 4
    row = 8 // col    
    fig, axs = plt.subplots(row, col, figsize=(col*2, row*2))


    for i in range(8):
        x = i % col
        y = i // col
        axs[y,x].imshow(level_images[i])
        axs[y,x].axis('off')
        axs[y,x].set_title(f'{levels[i]}')
    
    plt.show()


def decode_img(image):
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return tf.image.resize(image, [consts.IMAGE_SIZE, consts.IMAGE_SIZE])


@tf.function
def pre_process_image(file_path):
    image = tf.io.read_file(file_path)
    image = decode_img(image)

    return image, image

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=consts.BUFFER_SIZE)
    ds = ds.batch(consts.BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def repeat_images(file_path):
    return tf.data.Dataset.from_tensors(file_path).repeat(consts.DUPLICATE_IMAGES)

def load_datasets(val_percentage):
    print('Loading datasets...')

    folder_path = f'res/images/{consts.IMAGE_SIZE}'

    list_ds = tf.data.Dataset.list_files(str(f'{folder_path}/*'), shuffle=False)
    list_ds = list_ds.shuffle(list_ds.cardinality(), reshuffle_each_iteration=False)

    image_files = os.listdir(folder_path)
    image_count = len(image_files)

    val_size = int(image_count * val_percentage)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    print("Training Images: ", tf.data.experimental.cardinality(train_ds).numpy())
    print("Evaluation Images: ", tf.data.experimental.cardinality(val_ds).numpy())

    if consts.DUPLICATE_IMAGES > 1:
        train_ds = train_ds.repeat(consts.DUPLICATE_IMAGES)
        val_ds = val_ds.repeat(consts.DUPLICATE_IMAGES)
    
    # Immediately calculate the cardinality after duplication and before further transformations
    print("Training Images (post-duplication): ", tf.data.experimental.cardinality(train_ds).numpy())
    print("Validation Images (post-duplication): ", tf.data.experimental.cardinality(val_ds).numpy())

    train_ds = train_ds.map(lambda x: pre_process_image(x), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x: pre_process_image(x), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    return train_ds, val_ds

def load_examples():

    print('Loading examples...')

    folder_path = f'res/images/{consts.IMAGE_SIZE}'
    files = tf.data.Dataset.list_files(str(f'{folder_path}/*'), shuffle=False).take(4)

    images = []

    for file in files:
        image = tf.io.read_file(file)
        image = decode_img(image)
        
        images.append(image)

    return images
