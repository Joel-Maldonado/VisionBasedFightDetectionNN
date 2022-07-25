from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import os
import tensorflow_addons as tfa
from tensorflow.keras import layers

# GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  
print(tf.config.list_physical_devices())
strategy = tf.distribute.MirroredStrategy()

# Random seed to ensure reproducibility
SEED = 42
tf.random.set_seed(SEED)

# Constants
IMG_SIZE = 224
MAX_VIDEOS_PER_CLASS = 10
BATCH_SIZE = 32
CHANNELS = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATA_DIR = './ViolenceImg'
LABEL_DIR = './violence_bounding_box_2000_3.csv'

df = pd.read_csv(LABEL_DIR)
df['Path'] = df['ImageID'].apply(lambda x: os.path.join(DATA_DIR, x))


DATASET_SIZE = len(df)

# Processing Functions
@tf.function
def preprocess(image, labels):
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    h, w = image.shape[:2]
    # y1, x1, y2, x2 = labels[0], labels[1], labels[2], labels[3]
    
    # image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
    # image = tf.image.rgb_to_grayscale(image)
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

# image_ds = path_ds.map(load_and_preprocess_image)#.batch(MAX_FRAMES)
label_ds = tf.data.Dataset.from_tensor_slices(df.to_numpy())
# label_ds = label_ds.map(adjust_labels_for_pad)

image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))

image_label_ds = image_label_ds.map(load_and_preprocess_image)

@tf.function
def GIoU_MSE(y_true, y_pred):
    giou = tfa.losses.giou_loss(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    return giou + mse

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss_min_percentage=0.1, n_past_epochs=5):
        self.val_losses = []
        self.loss_min_percentage = loss_min_percentage
        self.n_past_epochs = n_past_epochs

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(float(logs['val_loss']))

        n_past_epochs = min(len(self.val_losses), self.n_past_epochs)
        average_slope = (float(logs['val_loss']) - self.val_losses[-n_past_epochs]) / 2

        if n_past_epochs < self.n_past_epochs:
            return

        # print(f"\nAverage slope for past {n_past_epochs} epochs: {average_slope}")

        if average_slope > -self.loss_min_percentage:
            print(f"\nSlope reached a slope of {average_slope} which is lower than loss_min_percentage, stopping training.")
            self.model.stop_training = True

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
    with strategy.scope():
        
        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(512, activation='relu'),
        #     tf.keras.layers.Dense(4, activation='sigmoid')
        # ])
        

        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.02, decay=, momentum=0.9, nesterov=True)
        NEURONS = 148
        DROPOUT = 0
        N_LAYERS = 6
        OPTIMIZER = tf.keras.optimizers.Nadam()

        def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
            return lr

        model = tf.keras.models.Sequential()

        model.add(layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))

        model.add(layers.Conv2D(NEURONS, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(DROPOUT))
        model.add(layers.BatchNormalization())
        
        for _ in range(N_LAYERS):
            model.add(layers.Conv2D(NEURONS, (3, 3), activation='relu', padding='same'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Dropout(DROPOUT))
            model.add(layers.BatchNormalization())

        model.add(layers.Flatten())

        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(DROPOUT))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Dense(4, activation='sigmoid')) 


        model.compile(
            loss=GIoU_MSE,
            optimizer=OPTIMIZER,
            metrics=['mse'],
        )

        return model


early_stopper = tf.keras.callbacks.EarlyStopping('val_accuracy', patience=20, restore_best_weights=True)

from datetime import datetime
time_date = datetime.now().strftime("%I-%M-%p")

check_point = tf.keras.callbacks.ModelCheckpoint(f'Checkpoints/violence_model_{time_date}.h5', save_best_only=True)
loss_percent_stopper = CustomCallback(gain_min_percentage=0.5, n_past_epochs=2)

model = create_model()

# history = model.fit(train_dataset, 
#                     validation_data=val_dataset, 
#                     epochs=5, 
#                     callbacks=[check_point, early_stopper], 
#                     use_multiprocessing=True, 
#                     workers=32,
#                     batch_size=BATCH_SIZE,
#                     )

# model.save(f'PersonDetection_temp.h5')


# metrics = model.evaluate(test_dataset)


# # if metrics[0] < 0.3:
# model.save(f'Models/Violence_Acc_{metrics[1]}.h5')