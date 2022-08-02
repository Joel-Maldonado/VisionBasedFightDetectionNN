# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()

import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import os
import tensorflow_addons as tfa
from tensorflow.keras import layers

# GPUs
# strategy = tf.distribute.MirroredStrategy()

# Random seed to ensure reproducibility
SEED = 42
tf.random.set_seed(SEED)

# Constants
IMG_SIZE = 224
MAX_VIDEOS_PER_CLASS = 10
BATCH_SIZE = 32
CHANNELS = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATA_DIR = 'Data/BoundingImages'
LABEL_DIR = 'Data/violence_bounding_box_labels.csv'

df = pd.read_csv(LABEL_DIR)
df['Path'] = df['ImageID'].apply(lambda x: os.path.join(DATA_DIR, x))


DATASET_SIZE = len(df)

# Processing Functions
@tf.function
def preprocess(image, labels):
    image = tf.image.decode_jpeg(image, channels=CHANNELS)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0

    return image, labels * [image.shape[0] / image.shape[1], 1, image.shape[0] / image.shape[1], 1]

@tf.function
def load_and_preprocess_image(img_path, labels):
    image = tf.io.read_file(img_path)
    return preprocess(image, labels)

@tf.function
def random_flip_horizontal(image, boxes):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[2], boxes[1], 1 - boxes[0], boxes[3]], axis=-1
        )
    return image, boxes

# Data
paths = df['Path']
df = df[['YMin','XMin', 'YMax', 'XMax']]

# Creating Dataset
path_ds = tf.data.Dataset.from_tensor_slices(paths.to_numpy())

label_ds = tf.data.Dataset.from_tensor_slices(df.to_numpy())

image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))

image_label_ds = image_label_ds.map(load_and_preprocess_image)

@tf.function
def GIoU_MSE(y_true, y_pred):
    giou = tfa.losses.giou_loss(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    return giou + mse

ds = image_label_ds.shuffle(buffer_size=DATASET_SIZE, seed=SEED)

# Train, test, val
train_size = int(0.75 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

train_dataset = ds.take(train_size)
test_dataset = ds.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

train_dataset.map(random_flip_horizontal, AUTOTUNE)

train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)


train_dataset.prefetch(AUTOTUNE)
test_dataset.prefetch(AUTOTUNE)
val_dataset.prefetch(AUTOTUNE)


# Model Creation
def create_model():
    NEURONS = 148
    DROPOUT = 0
    N_LAYERS = 2
    OPTIMIZER = tf.keras.optimizers.Nadam()


    model = tf.keras.models.Sequential()

    model.add(layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))

    model.add(layers.Conv2D(NEURONS, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    for _ in range(N_LAYERS):
        model.add(layers.Conv2D(NEURONS, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(DROPOUT))
    
    model.add(layers.Dense(4, activation='sigmoid')) 


    model.compile(
        loss=GIoU_MSE,
        optimizer=OPTIMIZER,
        metrics=['mse'],
    )

    return model


early_stopper = tf.keras.callbacks.EarlyStopping('val_loss', patience=20, restore_best_weights=True)

from datetime import datetime
time_date = datetime.now().strftime("%I-%M-%p")

check_point = tf.keras.callbacks.ModelCheckpoint(f'Checkpoints/bounding_box_{time_date}.h5', save_best_only=True)
reduce_lr_on_plataeu = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10)

# with strategy.scope():
model = create_model()

history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    epochs=200, 
                    callbacks=[check_point, early_stopper, reduce_lr_on_plataeu, tf.keras.callbacks.TensorBoard("tb_logs")], 
                    use_multiprocessing=True, 
                    workers=16,
                    batch_size=BATCH_SIZE)

metrics = model.evaluate(test_dataset)

model.save(f'Models/Bounding_Box_GIoU_{metrics[0]}.h5')

tf.keras.utils.plot_model(model, to_file='bounding_box_model.png', show_shapes=True)