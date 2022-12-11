# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()

import pandas as pd
import tensorflow as tf
import os
import tensorflow_addons as tfa
from tensorflow.keras import layers

# GPUs
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

# print(tf.config.list_physical_devices())
# strategy = tf.distribute.MirroredStrategy()

# Random seed to ensure reproducibility
SEED = 42
tf.random.set_seed(SEED)

# Constants
IMG_SIZE = 256
MAX_VIDEOS_PER_CLASS = 10
BATCH_SIZE = 32
CHANNELS = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE

DATA_DIR = 'Data/BinaryImages'
V_DIR = os.path.join(DATA_DIR, '1_violent')
NV_DIR = os.path.join(DATA_DIR, '0_non_violent')
DATASET_SIZE = len(os.listdir(NV_DIR)) + len(os.listdir(V_DIR))

@tf.function
def preprocess_image(img_path):
    raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(raw, channels=CHANNELS, expand_animations=False)
    # img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    img_predict = img / 255.0
    return img_predict

def make_dataset():
    nv_dataset = tf.data.Dataset.list_files(os.path.join(NV_DIR, '*.jpg'))
    nv_dataset = nv_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    nv_dataset = nv_dataset.map(lambda x: (x, tf.constant(0)))

    v_dataset = tf.data.Dataset.list_files(os.path.join(V_DIR, '*.jpg'))
    v_dataset = v_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    v_dataset = v_dataset.map(lambda x: (x, tf.constant(1)))

    combined_dataset = nv_dataset.concatenate(v_dataset)
    
    combined_dataset = combined_dataset.shuffle(DATASET_SIZE)

    return combined_dataset

dataset = make_dataset()

# # Train, test, val
train_size = int(0.75 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

print(train_dataset)

train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)


train_dataset = train_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)

train_dataset = train_dataset.cache()
val_dataset = val_dataset.cache()


# Model Creation
def create_model():

    NEURONS = 148
    DROPOUT = 0
    N_LAYERS = 2

    model = tf.keras.models.Sequential()

    model.add(layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))

    model.add(layers.RandomFlip())
    model.add(layers.RandomRotation(0.008))

    model.add(layers.Conv2D(NEURONS, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(DROPOUT))
    model.add(layers.BatchNormalization())
    
    for _ in range(N_LAYERS):
        model.add(layers.Conv2D(NEURONS, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(DROPOUT))
        model.add(layers.BatchNormalization())

    model.add(layers.Flatten())

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(DROPOUT))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(1, activation='sigmoid')) 


    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Nadam(0.007179202232340566),
        metrics=['accuracy'],
    )

    return model


early_stopper = tf.keras.callbacks.EarlyStopping('val_accuracy', patience=20, restore_best_weights=True)
reduce_lr_on_plataeu = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=10)

from datetime import datetime
time_date = datetime.now().strftime("%I-%M-%p")

check_point = tf.keras.callbacks.ModelCheckpoint(f'Checkpoints/violence_model_{time_date}.h5', save_best_only=True)
# loss_percent_stopper = CustomCallback(loss_min_percentage=0.02, n_past_epochs=20)

model = create_model()

history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    epochs=100, 
                    callbacks=[check_point, early_stopper, reduce_lr_on_plataeu], 
                    use_multiprocessing=True, 
                    workers=32,
                    batch_size=BATCH_SIZE,
                    )

model.save(f'PersonDetection_temp.h5')


metrics = model.evaluate(test_dataset)


# if metrics[0] < 0.3:
model.save(f'Models/Violence_Binary_Acc_{metrics[1]}.h5')