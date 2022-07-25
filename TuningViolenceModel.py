# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()

import pandas as pd
import tensorflow as tf
import os
import tensorflow_addons as tfa
from tensorflow.keras import layers

print(tf.config.list_physical_devices())
strategy = tf.distribute.MirroredStrategy()

# Random seed to ensure reproducibility
SEED = 42
tf.random.set_seed(SEED)

# Constants
IMG_SIZE = 224
N_FRAMES = 20
BATCH_SIZE = 8
CHANNELS = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE

def parse_tfrecord(example):
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'feature': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, features)
    video = tf.io.decode_raw(example['feature'], tf.float32)
    video = tf.reshape(video, (N_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS))
    label = example['label']
    return video, label

def preprocess(video, label):
    video = video / 255.0
    return video, label

train_dataset = tf.data.TFRecordDataset('Data/violence_video_train.tfrecord')
train_dataset = train_dataset.map(parse_tfrecord)
train_dataset = train_dataset.map(preprocess)


val_dataset = tf.data.TFRecordDataset('Data/violence_video_val.tfrecord')
val_dataset = val_dataset.map(parse_tfrecord)
val_dataset = val_dataset.map(preprocess)


print(train_dataset)

train_dataset = train_dataset.shuffle(buffer_size=1000)
val_dataset = val_dataset.shuffle(buffer_size=1000)

train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)


train_dataset.prefetch(AUTOTUNE)
val_dataset.prefetch(AUTOTUNE)



# Model Creation
def create_model():

    NEURONS = 148
    DROPOUT = 0.5
    N_LAYERS = 2

    model = tf.keras.models.Sequential()

    model.add(layers.ConvLSTM2D(
        32,
        kernel_size=(3, 3),
        return_sequences=False,
        input_shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(DROPOUT))

    model.add(layers.Conv2D(NEURONS, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(DROPOUT))

    model.add(layers.Conv2D(NEURONS, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(DROPOUT))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Nadam(0.001),
        metrics=['accuracy'],
    )

    return model


early_stopper = tf.keras.callbacks.EarlyStopping('val_accuracy', patience=20, restore_best_weights=True)
reduce_lr_on_plataeu = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=10)

from datetime import datetime
time_date = datetime.now().strftime("%I-%M-%p")

check_point = tf.keras.callbacks.ModelCheckpoint(f'Checkpoints/violence_model_{time_date}.h5', save_best_only=True)
# loss_percent_stopper = CustomCallback(loss_min_percentage=0.02, n_past_epochs=20)

with strategy.scope():
    model = create_model()

history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    epochs=50, 
                    callbacks=[
                        check_point, 
                        early_stopper, 
                        reduce_lr_on_plataeu], 
                        # tf.keras.callbacks.TensorBoard("tb_logs")], 
                    use_multiprocessing=True, 
                    workers=32,
                    batch_size=BATCH_SIZE,
                    )

model.save(f'PersonDetection_temp.h5')


metrics = model.evaluate(val_dataset)

# if metrics[0] < 0.3:
model.save(f'Models/Violence_Acc_{metrics[1]}.h5')