# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

# GPUs
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

print(tf.config.list_physical_devices())
strategy = tf.distribute.MirroredStrategy()

# Random seed to ensure reproducibility
SEED = 435
tf.random.set_seed(SEED)

# Constants
IMG_SIZE = 224
N_FRAMES = 20
BATCH_SIZE = 16
CHANNELS = 5
AUTOTUNE = tf.data.experimental.AUTOTUNE
ROTATION_MAX = 0.05

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

# class RandomBlurVideo(tf.keras.layers.Layer):
#   def __init__(self, max_shift=10, **kwargs):
#     super(RandomBlurVideo, self).__init__()
#     self.max_shift = max_shift

#   @tf.function
#   def call(self, inputs):
#     random_factor = tf.random.uniform(()) * self.max_shift + 1
#     return tf.map_fn(lambda x: tfa.image.gaussian_filter2d(x, (random_factor, random_factor)), inputs)
  
#   def get_config(self):
#     config = super().get_config().copy()
#     return config


# Model Creation
def create_model():
    global ROTATION_MAX

    CONV_NEURONS = 16
    DROPOUT = 0.2
    LR = 0.0007
    DENSE_UNITS = 64
    N_CONV_LAYERS = 3
    N_DENSE_LAYERS = 1
    STRIDES = 1
    ROTATION_MAX = 0.212

    model = tf.keras.models.Sequential()

    model.add(layers.InputLayer(input_shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS)))

    model.add(layers.Conv3D(CONV_NEURONS, (3, 3, 3), strides=STRIDES, activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(layers.Conv3D(CONV_NEURONS, (3, 3, 3), strides=STRIDES, activation='relu', kernel_initializer='he_normal', padding='same'))
    # model.add(layers.TimeDistributed(layers.Conv2D(CONV_NEURONS, (3, 3), strides=STRIDES, activation='relu', kernel_initializer='he_normal', padding='same')))
    # model.add(layers.TimeDistributed(layers.Conv2D(CONV_NEURONS, (3, 3), strides=STRIDES, activation='relu', kernel_initializer='he_normal', padding='same')))
    model.add(layers.MaxPooling3D())

    for i in range(N_CONV_LAYERS - 1):
      model.add(layers.Conv3D(CONV_NEURONS, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
      model.add(layers.Conv3D(CONV_NEURONS, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
      # model.add(layers.TimeDistributed(layers.Conv2D(CONV_NEURONS, (3, 3), strides=STRIDES, activation='relu', kernel_initializer='he_normal', padding='same')))
      # model.add(layers.TimeDistributed(layers.Conv2D(CONV_NEURONS, (3, 3), strides=STRIDES, activation='relu', kernel_initializer='he_normal', padding='same')))

      model.add(layers.MaxPooling3D())

    model.add(layers.Flatten())

    for i in range(N_DENSE_LAYERS):
      model.add(layers.Dense(DENSE_UNITS, activation='relu'))
      model.add(layers.Dropout(DROPOUT))


    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer= tf.keras.optimizers.Adam(LR),
        metrics=['accuracy'],
    )

    return model

import tensorflow_addons as tfa


early_stopper = tf.keras.callbacks.EarlyStopping('val_accuracy', patience=30, restore_best_weights=True)
reduce_lr_on_plataeu = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5)

from datetime import datetime
time_date = datetime.now().strftime("%I-%M-%p")

acc_cp = tf.keras.callbacks.ModelCheckpoint(f'Checkpoints/violence_model_acc_{time_date}.h5', save_best_only=True, monitor='val_accuracy')
loss_cp = tf.keras.callbacks.ModelCheckpoint(f'Checkpoints/violence_model_loss_{time_date}.h5', save_best_only=True, monitor='val_loss')

with strategy.scope():
    model = create_model()


tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, dpi=150, to_file='violence_model_hr.png', rankdir='LR')



# history = model.fit(train_dataset, 
#                     validation_data=val_dataset, 
#                     epochs=6, 
#                     callbacks=[acc_cp, loss_cp, early_stopper, reduce_lr_on_plataeu, tf.keras.callbacks.TensorBoard("tb_logs")], 
#                     use_multiprocessing=True, 
#                     workers=32,
#                     batch_size=BATCH_SIZE,
                    # )



# model.save(f'PersonDetection_temp.h5')

# metrics = model.evaluate(val_dataset)

# model.save(f'Models/Violence_Acc_{metrics[1]}.h5')


