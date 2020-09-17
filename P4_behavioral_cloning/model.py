#! Lint as: Python3

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behaviorial cloning models for self-driving car."""

import argparse
import collections
import csv
from datetime import datetime
import functools
import math
import os
import pickle
import sys
from typing import Any, Callable, Dict, List

try:
    from packaging import version
except ImportError:
    print('Please install package dependencies: pip install -r requirements.txt')
    sys.exit(1)

import cv2
import keras
from keras import callbacks as keras_callbacks
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
from sklearn import model_selection
from sklearn import utils as skutils
import tensorflow as tf

SEED_VALUE = 42
np.random.seed(SEED_VALUE)
tf.set_random_seed(SEED_VALUE)

# Input CSV header for reference
CSV_HEADER = ['center', 'left', 'right', 'steer', 'throttle', 'break', 'speed']

# Image dimensions
IMG_WIDTH = 320
IMG_HEIGHT = 160
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

MODELS_DIR = './models'
DATA_DIR = './data'

DATASETS = [
    #'run1',
    #'track1_redo',
    'track1_4laps', 
    'track1_reverse', 
    #'track1_recovery',
    'track1_centering',
    'track2_redo',
]

# Hyperparameters
HParams = collections.namedtuple('HParams', [
    'datasets', 
    'batch_size', 
    'learning_rate', 
    'num_epochs',
    'dropout_prob',
    'use_early_stopping',
    'early_stopping_patience',
    'train_steps',
    'valid_steps',
    'angle_correction',
    'augment_threshold',
    'clr_probs',
    'load_model',
    'throttle_threshold',
])
 

def timestamp() -> str:
    """Returns current timestamp.""" 
    return datetime.now().strftime("%Y%m%d_%H%M%S")
    

def normalize(img, minval=-1, maxval=1):
    """Returns normalized image."""
    
    #return img / 255. - .5
    old_shape = list(eval(str(img.shape[1:])))
    n = np.prod(old_shape)
    img_array = tf.cast(tf.reshape(img, [-1, n]), tf.float32)
    min_ = tf.expand_dims(tf.reduce_min(img_array, axis=1), -1)
    max_ = tf.expand_dims(tf.reduce_max(img_array, axis=1), -1)
    norm_array = minval + (maxval - minval)*(img_array - min_)/(max_ - min_)
    return tf.reshape(norm_array, [-1] + old_shape) 


def filter_samples_by_throttle(samples, throttle_threshold=0.):
    """Filter samples based on throttle value."""
    
    if not throttle_threshold:
        return samples
    
    out_samples = []
    throttle_idx = CSV_HEADER.index('throttle')
    for s in samples:
        throttle = float(s[throttle_idx])
        if throttle > throttle_threshold:
            out_samples.append(s)

    return out_samples


def load_data(
    datasets: List[str], validation_split=0.2, data_dir=DATA_DIR, filter_fn=None):
    """Loads multiple datasets"""
    
    samples = []
    for d in datasets:
        csv_file = os.path.join(data_dir, d, 'driving_log.csv')
        with open(csv_file) as fobj:
            reader = csv.reader(fobj)
            for line in reader:
                samples.append(line)
            
    train_samples, validation_samples = model_selection.train_test_split(
        samples, test_size=validation_split)
    
    if filter_fn:
        train_samples = filter_fn(train_samples)

    return train_samples, validation_samples


def change_image_path(image_path, base_dir):
    return os.path.join(base_dir, os.path.split(image_path)[-1])


def load_image(image_path: str, colorspace: int = None):
    """Load image from file path."""
    
    img = cv2.imread(image_path)
    if colorspace:
        img = cv2.cvtColor(img, colorspace)
    
    return img


def random_shift(img, angle, max_x_shift=100, max_y_shift=40, shift_angle_factor=.4):
    """Randomly shift image in x and y direction."""
    
    x_shift = max_x_shift*np.random.uniform() - max_x_shift/2
    shift_angle = angle + x_shift/max_x_shift*shift_angle_factor
    y_shift = max_y_shift*np.random.uniform() - max_y_shift/2
    M_shift = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    rows, cols = img.shape[0], img.shape[1]
    shift_img = cv2.warpAffine(img, M_shift, (cols, rows))
    return shift_img, shift_angle
    

def fliplr(img: np.array, angle: float):
    """Flips image horizontally and negates steering angle accordingly."""
    
    flip_img = np.fliplr(img)
    return flip_img, -angle


def adjust_brightness(hsv_img, value: float = 1.):
    out_img = hsv_img.astype(np.float64)
    out_img[:, :, 2] = np.clip(out_img[:, :, 2]*value, 0, 255)
    return out_img.astype(np.uint8)


def random_brightness(hsv_img, retval=False):
    value = .5 + np.random.uniform()
    adj_img = adjust_brightness(hsv_img, value)
    if retval:
      return adj_img, value

    return adj_img


def random_shadow(hsv_img):
    """Adds random shadow.

    Reference: https://markku.ai/post/data-augmentation/
    """
    out_img = np.copy(hsv_img).astype(np.float64)
    height= out_img.shape[0]
    width = out_img.shape[1]
    top_x = width * np.random.uniform()
    top_y = 0
    bot_x = width * np.random.uniform()
    bot_y = height
    shadow_mask = np.zeros_like(out_img[:, :, 1])

    m = np.mgrid[0:height, 0:width]
    X_m, Y_m = m[0], m[1]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) > 0)] = 1

    shadow_density = .5
    left_side = shadow_mask == 1
    right_side = shadow_mask == 0

    if np.random.randint(2) == 1:
        out_img[:, :, 2][left_side] *= shadow_density
    else:
        out_img[:, :, 2][right_side] *= shadow_density

    return out_img.astype(np.uint8)


def subsample(func, samples_per_epoch, angle_threshold=0.1, max_iter=1000):
    """Decorator for subsampling training data."""

    sample_i = 0
    epoch = 1
    keep_threshold = 1 / (epoch + 1)
    
    def wrapper(*args, **kwargs):
        nonlocal keep_threshold
        for _ in range(max_iter):
            img, angle = func(*args, **kwargs)
            if abs(angle) < angle_threshold:
                prob = np.random.uniform()
                if prob > keep_threshold:
                    break
            else:
                break
                
        nonlocal sample_i
        nonlocal epoch
        sample_i += 1
        if sample_i > samples_per_epoch:
            sample_i = 0
            epoch += 1
            keep_threshold = 1 / (epoch + 1)
        
        #print(sample_i, samples_per_epoch, keep_threshold)
        return img, angle
    
    return wrapper

def preprocess_prediction_sample(img, from_bgr=False):
    """Preprocess image for prediction."""
   
    colorspace = cv2.COLOR_BGR2YUV if from_bgr else cv2.COLOR_RGB2YUV
    return cv2.cvtColor(img, colorspace)


def preprocess_training_sample(
    sample: List[Any], angle_correction=0.2, augment_threshold=0.4, clr_probs=[0.34, 0.33, 0.33]):
    """Preprocess a sample (image, angle) for training dataset."""
    
    # Randomly pick between center, left or right images
    img_idx = np.random.choice(3, p=clr_probs)
    img_path = sample[img_idx]
    angle = float(sample[3])
    if img_idx == 1:  # left camera
        angle += angle_correction
    elif img_idx == 2:  # right camera
        angle -= angle_correction

    prob = np.random.uniform()
    if prob > augment_threshold:
        hsv_img = load_image(img_path, cv2.COLOR_BGR2HSV)
        
        # Random brightness
        hsv_img = random_brightness(hsv_img)

        # Random shadow
        hsv_img = random_shadow(hsv_img)

        # Random shift
        #hsv_img, angle = random_shift(hsv_img, angle)

        # Randomly flip the image
        do_flip = np.random.choice(2)
        if do_flip:
            hsv_img, angle = fliplr(hsv_img, angle)
        
        img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    else:
        img = load_image(img_path)
        
    img = preprocess_prediction_sample(img, from_bgr=True)
    return img, angle


def preprocess_validation_sample(sample: List[Any]):
    """Preprocess a sample (image, angle) for validation dataset."""
    img = load_image(sample[0])
    img = preprocess_prediction_sample(img, from_bgr=True)
    angle = float(sample[3])
    return img, angle


def make_generator(
        samples, batch_size=1, image_dir=None, preprocess_fn=None):
    """Makes generator for reading data."""
    
    num_samples = len(samples)
    if not preprocess_fn:
        preprocess_fn = lambda x: x
    while True: # Loop forever so the generator never terminates
        skutils.shuffle(samples)
        for i, offset in enumerate(range(0, num_samples, batch_size)):
            batch_samples = samples[offset: offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                img, angle = preprocess_fn(batch_sample)
                images.append(img)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield skutils.shuffle(X_train, y_train)


def get_simple_dnn(input_shape, hp: HParams):
    """Returns a simple DNN model for debugging."""
    
    _ = hp  # hparams not needed
    model = models.Sequential(name='SimpleDNN')
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(1))
    return model


def conv2d(filters, kernel_size, strides=1, activation=None, suffix=None):
    """Keras Conv2D wrapper with custom name convention."""
    name='conv2d_f{}_k{}_s{}_{}'.format(
        filters, kernel_size, strides, activation)
    if suffix:
      name += '_{}'.format(suffix)
    return layers.Conv2D(
        filters=filters, 
        kernel_size=kernel_size, 
        strides=strides, 
        activation=activation,
        name=name)
  

def cropping2d(cropping, suffix=None):
    """Keras Cropping2D wrapper with custom name convention."""
    (top, bottom), (left, right) = cropping
    name = 'cropping2d_t{}_b{}_l{}_r{}'.format(
        top, bottom, left, right)
    if suffix:
        name += '_{}'.format(suffix)
    return layers.Cropping2D(cropping=cropping, name=name)

  
def get_nvidia_cnn(input_shape, hp: HParams):
    """Returns a modified CNN from the Nvidia paper (ArXiv ID 1604.07316v1)."""
    
    inp = layers.Input(shape=input_shape, name='yuv_input')
    
    # This crops the image to only see center of the road
    top_crop = int(input_shape[0] * 0.2)
    x = cropping2d(cropping=((top_crop, 25), (0, 0)))(inp)
    
    # Resize
    x = layers.Lambda(lambda img: tf.image.resize_images(img, (66, 200)), name='resize')(x)
    
    x = layers.Lambda(normalize, name='normalize')(x)
    x = conv2d(filters=24, kernel_size=5, strides=2, activation='relu')(x)
    x = conv2d(filters=36, kernel_size=5, strides=2, activation='relu')(x)
    x = conv2d(filters=48, kernel_size=5, strides=2, activation='relu')(x)
    x = conv2d(filters=64, kernel_size=3, strides=1, activation='relu')(x)
    x = conv2d(filters=64, kernel_size=3, strides=1, activation='relu', suffix='2')(x)
    #x = layers.Dropout(rate=hp.dropout_prob, seed=SEED_VALUE)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=hp.dropout_prob, seed=SEED_VALUE)(x)
    x = layers.Dense(1164, activation='relu')(x)
    x = layers.Dropout(rate=hp.dropout_prob, seed=SEED_VALUE)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(rate=hp.dropout_prob, seed=SEED_VALUE)(x)
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dense(10, activation='relu')(x)
    out = layers.Dense(1)(x)
    keras_model = models.Model(inp, out, name='NvidiaCNN')

    return keras_model


def get_augment_cnn(input_shape, hp: HParams):
    """CNN from the work of Vivek Yadav.
    
    Reference:
    https://github.com/vxy10/P3-BehaviorCloning
    """
    
    inp = layers.Input(shape=input_shape)
    
    # This trims the image to only see center of the road
    top_crop = int(input_shape[0] * 0.2)
    x = layers.Cropping2D(cropping=((top_crop, 25), (0, 0)))(inp)
    
    x = layers.Lambda(lambda img: tf.image.resize_images(img, (64, 64)), name='resize')(x)
    x = layers.Lambda(normalize, name='normalize')(x)
    
    # Colorspace conversion
    x = layers.Conv2D(filters=3, kernel_size=1, strides=1)(x)
    
    activation = 'elu'
    kernel_size = 3
    pool_size = 2
    x = layers.Conv2D(filters=32, kernel_size=kernel_size, strides=1, activation=activation)(x)
    x = layers.Conv2D(filters=32, kernel_size=kernel_size, strides=1, activation=activation)(x)
    x = layers.MaxPooling2D(pool_size=pool_size)(x)
    x = layers.Dropout(rate=hp.dropout_prob, seed=SEED_VALUE)(x)
    
    x = layers.Conv2D(filters=64, kernel_size=kernel_size, strides=1, activation=activation)(x)
    x = layers.Conv2D(filters=64, kernel_size=kernel_size, strides=1, activation=activation)(x)
    x = layers.MaxPooling2D(pool_size=pool_size)(x)
    x = layers.Dropout(rate=hp.dropout_prob, seed=SEED_VALUE)(x)
    
    x = layers.Conv2D(filters=128, kernel_size=kernel_size, strides=1, activation=activation)(x)
    x = layers.Conv2D(filters=128, kernel_size=kernel_size, strides=1, activation=activation)(x)
    x = layers.MaxPooling2D(pool_size=pool_size)(x)
    x = layers.Dropout(rate=hp.dropout_prob, seed=SEED_VALUE)(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(512, activation=activation)(x)
    x = layers.Dropout(rate=hp.dropout_prob, seed=SEED_VALUE)(x)
    x = layers.Dense(64, activation=activation)(x)
    x = layers.Dropout(rate=hp.dropout_prob, seed=SEED_VALUE)(x)
    x = layers.Dense(16, activation=activation)(x)
    x = layers.Dropout(rate=hp.dropout_prob, seed=SEED_VALUE)(x)
    out = layers.Dense(1)(x)
    keras_model = models.Model(inp, out, name='AugmentCNN')

    return keras_model
    

def print_training_summary(saved_model_name, history, hp: HParams, delimiter=','):
    """Prints CSV line summary of model training."""
    
    model_name = saved_model_name.split('_')[0]
    history = history.history
    train_loss, valid_loss = history['loss'][-1], history['val_loss'][-1]
    train_loss_str = '{:.4f}'.format(train_loss)
    valid_loss_str = '{:.4f}'.format(valid_loss)
    loss_ratio_str = '{:.2f}'.format(valid_loss / train_loss)
    
    def _stringify(v):
        float_format_fn = lambda x: '{:.4f}'.format(x)
        if isinstance(v, float):
            return float_format_fn(v)
        elif isinstance(v, list):
            return ':'.join([_stringify(vi) for vi in v])
        return str(v)
        
    summary = (
        [saved_model_name, model_name] +
        [_stringify(v) for v in hp] +
        [_stringify(v) for v in (
            train_loss_str, valid_loss_str, loss_ratio_str)])
    print(delimiter.join([str(x) for x in summary]))
    
    
def get_saved_model_name(model_name):
    return '{model_name}_{timestamp}'.format(
        model_name=model_name, timestamp=timestamp())


def save_model(keras_model: models.Model, history: Dict, hp: HParams):
    """Saves model, training history and hyperparameters."""
    
    # Save trained model
    saved_model_name = get_saved_model_name(keras_model.name)
    model_dir = os.path.join(MODELS_DIR, saved_model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, 'model.h5')
    keras_model.save(model_file)
    
    # Save training history and hparams
    history_file = os.path.join(MODELS_DIR, saved_model_name, 'context.pkl')
    with open(history_file, 'wb') as f:
        pickle.dump(dict(history=history.history, hparams=dict(hp._asdict())), f)

    # Print summary
    print_training_summary(saved_model_name, history, hp)
    print('Model saved as {}'.format(model_file))
    

def train_and_evaluate(keras_model, train_samples, validation_samples, hp: HParams):
    """Runs model training and evaluation."""
    
    # Preprocess dataset into generators
    generator_fn = functools.partial(
        make_generator, batch_size=hp.batch_size)
    preprocess_training_fn = functools.partial(
        preprocess_training_sample,
        angle_correction=hp.angle_correction,
        augment_threshold=hp.augment_threshold,
        clr_probs=hp.clr_probs)
    train_samples_per_epoch = hp.train_steps * hp.batch_size
    # Note: Disable subsampling as it doesn't improve performance
    #preprocess_training_fn = subsample(
    #    preprocess_training_fn, train_samples_per_epoch)
    train_generator = generator_fn(
        train_samples, preprocess_fn=preprocess_training_fn)
    validation_generator = generator_fn(
        validation_samples, preprocess_fn=preprocess_validation_sample)
    
    # Add callbacks
    callbacks = []
    if hp.use_early_stopping:
        print('Adding early stopping callback.')
        early_stopping_fn = functools.partial(
            keras_callbacks.EarlyStopping, 
            patience=hp.early_stopping_patience, 
            verbose=1)
        if version.parse(keras.__version__) <= version.parse('2.0.9'):
            callbacks.append(early_stopping_fn())
        else:
            callbacks.append(early_stopping_fn(restore_best_weights=True))

    # Do training
    history = keras_model.fit_generator(
        train_generator,
        steps_per_epoch=hp.train_steps,
        validation_data=validation_generator,
        validation_steps=hp.valid_steps,
        epochs=hp.num_epochs, 
        verbose=1,
        callbacks=callbacks,
    )

    save_model(keras_model, history, hp)


def parse_args(argv):
    """Parses command line arguments."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='NvidiaCNN', help='Model architecture to use.')
    parser.add_argument('-n', '--num_epochs', default=10, type=int, help='Number of training epochs.')
    parser.add_argument('-d', '--datasets', default=DATASETS, nargs='+', help='Datasets to use.')
    parser.add_argument('-t', '--train_steps', default=None, type=int, help='Training steps.')
    parser.add_argument('-v', '--valid_steps', default=None, type=int, help='Validation steps.')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='Train/valid batch size.')
    parser.add_argument('-l', '--learning_rate', default=0.001, type=float, help='Learning rate.')
    parser.add_argument('-p', '--dropout_prob', default=0., type=float, help='Dropout: probability of dropping nodes')
    parser.add_argument('--no_early_stopping', default=True, action='store_false', help='Disable early stopping')
    parser.add_argument('--angle_correction', default=0.2, type=float, help='Absolute angle correction for left/right camera images')
    parser.add_argument('--augment_threshold', default=0.4, type=float, help='Probability threshold for applying train image augmentation')
    parser.add_argument('--clr_probs', default=[0.34, 0.33, 0.33], type=float, nargs=3, help='Probabilities for choosing center/left/right camera images.')
    parser.add_argument('--load_model', default=None, help='Load model weights from saved model file.')
    parser.add_argument('--throttle_threshold', default=0., type=float,
                        help='Filter training samples by throttle exceeding this threshold.')           
    return parser.parse_args(argv)


_models = {
    'SimpleDNN': get_simple_dnn,
    'NvidiaCNN': get_nvidia_cnn,
    'AugmentCNN': get_augment_cnn,
}


def main():
    args = parse_args(sys.argv[1:])
    model_fn = _models.get(args.model)
    if not model_fn:
        print('Unsupported model: {}'.format(args.model))
        
    # Load dataset
    train_samples, validation_samples = load_data(
        args.datasets, 
        data_dir=DATA_DIR,
        filter_fn=functools.partial(
            filter_samples_by_throttle, 
            throttle_threshold=args.throttle_threshold),
    )
    n_train = len(train_samples)
    n_valid = len(validation_samples)
    print('Training samples: {}, validation samples: {}'.format(n_train, n_valid))
    
    # Set train/validation steps
    train_steps = args.train_steps or math.ceil(n_train / args.batch_size)
    #train_steps = args.train_steps or math.ceil(20000 / args.batch_size)
    valid_steps = args.valid_steps or math.ceil(n_valid / args.batch_size)
    
    assert len(args.clr_probs) == 3

    # Set hyperparameters
    hp = HParams(
        datasets=args.datasets,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        dropout_prob=args.dropout_prob,
        use_early_stopping=args.no_early_stopping,
        early_stopping_patience=3,
        train_steps=train_steps,
        valid_steps=valid_steps,
        angle_correction=args.angle_correction,
        augment_threshold=args.augment_threshold,
        clr_probs=args.clr_probs,
        load_model=args.load_model,
        throttle_threshold=args.throttle_threshold,
    )

    keras_model = model_fn(INPUT_SHAPE, hp)
    if args.load_model:
        model_file = os.path.join('models', args.load_model, 'model.h5')
        print('Loading saved model from: {}'.format(model_file))
        keras_model.load_weights(model_file)
    keras_model.compile(
        loss='mse', 
        optimizer=optimizers.Adam(hp.learning_rate))
    train_and_evaluate(keras_model, train_samples, validation_samples, hp)
    

if __name__ == '__main__':
    main()
