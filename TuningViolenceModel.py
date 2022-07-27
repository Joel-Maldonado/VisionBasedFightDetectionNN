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

# Random seed to ensure reproducibility
SEED = 24
tf.random.set_seed(SEED)

# Constants
IMG_SIZE = 224
N_FRAMES = 20
BATCH_SIZE = 16
CHANNELS = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE

TRAIN_RECORD_DIR = 'violence_video_train.tfrecord'
VAL_RECORD_DIR = 'violence_video_test.tfrecord'

def parse_tfrecord(example):
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'feature': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, features)
    video = tf.io.decode_raw(example['feature'], tf.float32)
    video = tf.reshape(video, (N_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS))
    label = tf.cast(example['label'], tf.uint8)
    return video, label

def preprocess(video, label):
    video = video / 255.0
    return video, label

train_dataset = tf.data.TFRecordDataset(TRAIN_RECORD_DIR)
train_dataset = train_dataset.map(parse_tfrecord)
train_dataset = train_dataset.map(preprocess)


val_dataset = tf.data.TFRecordDataset(VAL_RECORD_DIR)
val_dataset = val_dataset.map(parse_tfrecord)
val_dataset = val_dataset.map(preprocess)


DATASET_SIZE = 2300

train_dataset = train_dataset.shuffle(buffer_size=1840, reshuffle_each_iteration=True)
val_dataset = val_dataset.shuffle(buffer_size=460, reshuffle_each_iteration=True)

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

  def call(self, inputs):
    return tf.map_fn(self.rotate, inputs)
    
  def rotate(self, video):
    random_factor = self.max_rotation * self.max_rotation * 2 - self.max_rotation
    return tfa.image.rotate(video, random_factor)
  
  def get_config(self):
    config = super().get_config().copy()
    return config

# Model Creation
def create_model(hp):
  with strategy.scope():
    
    CONV_NEURONS = hp.Int('conv_neurons', 8, 40)
    DROPOUT = hp.Float('dropout', 0.0, 0.5)
    LR = hp.Float('lr', 0.0001, 0.006)
    ROTATION_MAX = hp.Float('rotation_max', 0.0, 0.5)
    DENSE_UNITS = hp.Int('dense_units', 0, 128)
    N_CONV_LAYERS = hp.Int('n_conv_layers', 2, 5)
    N_DENSE_LAYERS = hp.Int('n_dense_layers', 0, 4)
    BATCH_NORM = hp.Int('batch_norm', 0, 1)
    KERNEL_INITIALIZER = hp.Choice('kernel_initializer', ['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal'])

    optimizers = {
      'adam': tf.keras.optimizers.Adam(LR),
      'nadam': tf.keras.optimizers.Nadam(LR),
      'sgd': tf.keras.optimizers.SGD(LR, momentum=0.9, nesterov=True, decay=1e-6),
    }
    optimizer_str = hp.Choice('optimizer', ['adam', 'nadam', 'sgd'])

    model = tf.keras.models.Sequential()


    model.add(layers.InputLayer(input_shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS)))

    model.add(RandomFlipVideo())
    model.add(RandomRotationVideo(ROTATION_MAX))

    for i in range(N_CONV_LAYERS):
      model.add(layers.TimeDistributed(layers.Conv2D(CONV_NEURONS, (3, 3), kernel_initializer=KERNEL_INITIALIZER, activation='relu')))

      if BATCH_NORM:
        model.add(layers.BatchNormalization())

      model.add(layers.TimeDistributed(layers.Conv2D(CONV_NEURONS, (3, 3), kernel_initializer=KERNEL_INITIALIZER, activation='relu')))
      
      if BATCH_NORM:
        model.add(layers.BatchNormalization())
        
      model.add(layers.TimeDistributed(layers.MaxPooling2D()))

    model.add(layers.Flatten())
    
    for i in range(N_DENSE_LAYERS):
      model.add(layers.Dense(DENSE_UNITS, activation='relu'))
      if BATCH_NORM:
        model.add(layers.BatchNormalization())
      
      model.add(layers.Dropout(DROPOUT))

    model.add(layers.Dense(1, activation='sigmoid'))



    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers[optimizer_str],
        metrics=['accuracy'],
    )

    return model


early_stopper = tf.keras.callbacks.EarlyStopping('val_accuracy', patience=8, restore_best_weights=True)
reduce_lr_on_plataeu = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5)


tuner = BayesianOptimization(
  create_model,
  objective='val_accuracy',
  max_trials=100,
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

print(tuner.results_summary(100))

import time

sys.stdout = open(f'tuner_results_{time.time()}.txt', 'w')
print(tuner.results_summary(100))
sys.stdout.close()