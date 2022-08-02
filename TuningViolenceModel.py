# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()

import tensorflow as tf
import os
import tensorflow_addons as tfa
from tensorflow.keras import layers
from keras_tuner import BayesianOptimization
import sys

# GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

print(tf.config.list_physical_devices())
strategy = tf.distribute.MirroredStrategy()

SEED = 435
tf.random.set_seed(SEED)

# Constants
IMG_SIZE = 224
N_FRAMES = 20
BATCH_SIZE = 16
CHANNELS = 5
AUTOTUNE = tf.data.experimental.AUTOTUNE
ROTATION_MAX = 0.0
REDUCE_QUALITY = False


TRAIN_RECORD_DIR = 'violence_rgb_opt_train.tfrecord'
VAL_RECORD_DIR = 'violence_rgb_opt_val.tfrecord'

def parse_tfrecord(example):
  features = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'video': tf.io.FixedLenFeature([], tf.string)
  }
  example = tf.io.parse_single_example(example, features)
  # video = tf.io.decode_raw(example['video'], tf.uint8)
  video = tf.io.decode_raw(example['video'], tf.float32)
  video = tf.reshape(video, (N_FRAMES, IMG_SIZE, IMG_SIZE, 5))
  label = tf.cast(example['label'], tf.uint8)
  return video, label

def preprocess(video, label):
    video = video / 255.0
    return video, label

def random_flip(video, label):
  if tf.random.uniform(()) > 0.5:
    return tf.map_fn(lambda x: tf.image.flip_left_right(x), video), label
  return video, label

def random_rotation(video, label):
  random_factor = tf.random.uniform(()) * ROTATION_MAX * 2 - ROTATION_MAX
  return tf.map_fn(lambda x: tfa.image.rotate(x, random_factor), video), label
  
def random_reduce_quality(video, label):
  if REDUCE_QUALITY == False:
    return video, label
  factor = tf.random.uniform(()) * 1.5
  return tf.map_fn(
    lambda x: tf.image.resize(tf.image.resize(x, (int(IMG_SIZE / factor), int(IMG_SIZE / factor))), (IMG_SIZE, IMG_SIZE)), video), label

train_dataset = tf.data.TFRecordDataset(TRAIN_RECORD_DIR)
train_dataset = train_dataset.map(parse_tfrecord)
train_dataset = train_dataset.map(preprocess)
train_dataset = train_dataset.map(random_flip)
train_dataset = train_dataset.map(random_rotation)
train_dataset = train_dataset.map(random_reduce_quality)

val_dataset = tf.data.TFRecordDataset(VAL_RECORD_DIR)
val_dataset = val_dataset.map(parse_tfrecord)
val_dataset = val_dataset.map(preprocess)

train_dataset = train_dataset.shuffle(buffer_size=1600, reshuffle_each_iteration=True)


train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)


train_dataset = train_dataset.prefetch(AUTOTUNE)

class RandomFlipVideo(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(RandomFlipVideo, self).__init__()

  @tf.function
  def call(self, inputs):
    if tf.random.uniform(()) > 0.5:
      return tf.map_fn(lambda x: tf.image.flip_left_right(x), inputs)
    return inputs
  
class RandomRotationVideo(tf.keras.layers.Layer):
  def __init__(self, max_rotation=0.3, **kwargs):
    super(RandomRotationVideo, self).__init__()
    self.max_rotation = max_rotation

  @tf.function
  def call(self, inputs):
    random_factor = tf.random.uniform(()) * self.max_rotation * 2 - self.max_rotation
    return tf.map_fn(lambda x: tfa.image.rotate(x, random_factor), inputs)
    
  def get_config(self):
    config = super().get_config().copy()
    return config

class RandomBlurVideo(tf.keras.layers.Layer):
  def __init__(self, max_shift=10, **kwargs):
    super(RandomBlurVideo, self).__init__()
    self.max_shift = max_shift

  @tf.function
  def call(self, inputs):
    random_factor = tf.random.uniform(()) * self.max_shift + 1
    return tf.map_fn(lambda x: tfa.image.gaussian_filter2d(x, (random_factor, random_factor)), inputs)
  
  def get_config(self):
    config = super().get_config().copy()
    return config

# Model Creation
def create_model(hp):
  with strategy.scope():
    global ROTATION_MAX, REDUCE_QUALITY

    CONV_NEURONS = hp.Int('conv_neurons', 4, 64)
    DROPOUT = hp.Float('dropout', 0.0, 0.5)
    LR = hp.Float('lr', 0.0001, 0.006)
    ROTATION_MAX = hp.Float('rotation_max', 0.0, 0.5)
    DENSE_UNITS = hp.Int('dense_units', 4, 128)
    N_CONV_LAYERS = hp.Int('n_conv_layers', 2, 4)
    N_DENSE_LAYERS = 1
    STRIDES = 1
    BATCH_NORM = hp.Boolean('batch_norm?')
    DENSE_BATCH_NORM_MODE = hp.Int('dense_batch_norm_mode', 0, 2)
    REDUCE_QUALITY = hp.Boolean('reduce_quality?')

    optimizers = {
      'adam': tf.keras.optimizers.Adam(LR),
      'nadam': tf.keras.optimizers.Nadam(LR),
      'sgd': tf.keras.optimizers.SGD(LR, momentum=0.9, nesterov=True, decay=1e-6),
    }
    optimizer_str = hp.Choice('optimizer', ['adam', 'nadam', 'sgd'])

    model = tf.keras.models.Sequential()


    model.add(layers.InputLayer(input_shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS)))

    # Add image augmentation layers 
    # model.add(RandomFlipVideo())
    # model.add(RandomRotationVideo(ROTATION_MAX))
    # model.add(RandomBlurVideo(10))
    
    model.add(layers.Conv3D(CONV_NEURONS, (3, 3, 3), strides=STRIDES, activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(layers.Conv3D(CONV_NEURONS, (3, 3, 3), strides=STRIDES, activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(layers.MaxPooling3D())
    if BATCH_NORM:
        model.add(layers.BatchNormalization())

    for i in range(N_CONV_LAYERS - 1):
      model.add(layers.Conv3D(CONV_NEURONS, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
      model.add(layers.Conv3D(CONV_NEURONS, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
      model.add(layers.MaxPooling3D())
      if BATCH_NORM:
        model.add(layers.BatchNormalization())

    model.add(layers.Flatten())

    for i in range(N_DENSE_LAYERS):
      model.add(layers.Dense(DENSE_UNITS, activation='relu'))
      model.add(layers.Dropout(DROPOUT))
      if DENSE_BATCH_NORM_MODE == 0:
        model.add(layers.BatchNormalization())
      elif DENSE_BATCH_NORM_MODE == 1:
        model.add(layers.Dropout(DROPOUT))
      else:
        model.add(layers.Dropout(DROPOUT))
        model.add(layers.BatchNormalization())


    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer= optimizers[optimizer_str],
        metrics=['accuracy'],
    )

    return model


early_stopper = tf.keras.callbacks.EarlyStopping('val_accuracy', patience=8, restore_best_weights=True)
reduce_lr_on_plataeu = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5)


tuner = BayesianOptimization(
  create_model,
  objective='val_accuracy',
  max_trials=25,
  overwrite=True,
)

tuner.search(
  train_dataset, 
  validation_data=val_dataset, 
  epochs=20,
  callbacks=[early_stopper, reduce_lr_on_plataeu], 
  use_multiprocessing=True, 
  workers=32,
  batch_size=BATCH_SIZE,
)

print(tuner.results_summary(400))

import time

sys.stdout = open(f'tuner_results_{time.time()}.txt', 'w')
print(tuner.results_summary(400))
sys.stdout.close()