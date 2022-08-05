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
ROTATION_MAX = 0.1
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
  factor = tf.random.uniform(()) * 1.5
  return tf.map_fn(
    lambda x: tf.image.resize(tf.image.resize(x, (int(IMG_SIZE / factor), int(IMG_SIZE / factor))), (IMG_SIZE, IMG_SIZE)), video), label

train_dataset = tf.data.TFRecordDataset(TRAIN_RECORD_DIR)
train_dataset = train_dataset.map(parse_tfrecord)
train_dataset = train_dataset.map(preprocess)
train_dataset = train_dataset.map(random_flip)
train_dataset = train_dataset.map(random_rotation)

val_dataset = tf.data.TFRecordDataset(VAL_RECORD_DIR)
val_dataset = val_dataset.map(parse_tfrecord)
val_dataset = val_dataset.map(preprocess)

train_dataset = train_dataset.shuffle(buffer_size=1600, reshuffle_each_iteration=True)


train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)


train_dataset = train_dataset.prefetch(AUTOTUNE)


# Model Creation
def create_model(hp):
  with strategy.scope():
    global ROTATION_MAX

    DROPOUT = hp.Float('dropout', 0.0, 0.5)
    LR = hp.Float('lr', 0.0001, 0.01)
    ROTATION_MAX = hp.Float('rotation_max', 0.0, 0.5)
    STRIDES = 1
    FLAT_POOL = hp.Choice('flat_pool', ['avg', 'max', 'flatten'])

    optimizers = {
      'adam': tf.keras.optimizers.Adam(LR),
      'nadam': tf.keras.optimizers.Nadam(LR),
      'sgd': tf.keras.optimizers.SGD(LR, momentum=0.9, nesterov=True, decay=1e-6),
    }
    optimizer_str = hp.Choice('optimizer', ['adam', 'nadam', 'sgd'])


    model = tf.keras.models.Sequential()

    model.add(layers.InputLayer(input_shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS)))

    model.add(layers.Conv3D(64, 3, strides=STRIDES, padding='same', activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(64, 3, strides=STRIDES, padding='same', activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(128, 3, strides=STRIDES, padding='same', activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(256, 3, strides=STRIDES, padding='same', activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=2))
    model.add(layers.BatchNormalization())

    if FLAT_POOL == 'avg':
      model.add(layers.GlobalAveragePooling3D())
    elif FLAT_POOL == 'max':
      model.add(layers.GlobalMaxPooling3D())
    else:
      model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(DROPOUT))

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
  max_trials=32,
  overwrite=True,
)

tuner.search(
  train_dataset, 
  validation_data=val_dataset, 
  epochs=25,
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