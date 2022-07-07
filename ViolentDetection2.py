from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  
strategy = tf.distribute.MirroredStrategy()

# Random seed to ensure reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Constants
IMG_SIZE = 256
MAX_VIDEOS_PER_CLASS = 10
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
# DATA_DIR = '/u/jruiz_intern/Desktop/jruiz/Datasets/CustomViolence'
DATA_DIR = '/u/jruiz_intern/Desktop/jruiz/Datasets/RWF_Img'
CLASSES = sorted(os.listdir(DATA_DIR))

image_paths = []
for category in CLASSES:
    class_path = os.path.join(DATA_DIR, category)
    for img_path in os.listdir(class_path):
        print(img_path)
        image_paths.append([os.path.join(class_path, img_path), CLASSES.index(category)])

df = pd.DataFrame(image_paths, columns=['path', 'violent'])

DATASET_SIZE = len(df)

# Processing Functions
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image /= 255.0

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def test(idk):
    for i in idk:
        print(i)
    idk.map(load_and_preprocess_image)

# Data
print(df.head())

paths = df.drop('violent', axis=1)
labels = df['violent']

# Creating Dataset
path_ds = tf.data.Dataset.from_tensor_slices(paths.to_numpy())

image_ds = path_ds.unbatch().map(load_and_preprocess_image)#.batch(MAX_FRAMES)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels.to_numpy(), tf.int8))

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = image_label_ds.shuffle(buffer_size=DATASET_SIZE, seed=SEED)
print(ds)

# Train, test, val
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

train_dataset = ds.take(train_size)
test_dataset = ds.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)

print(train_dataset)

train_dataset.prefetch(AUTOTUNE)
test_dataset.prefetch(AUTOTUNE)
val_dataset.prefetch(AUTOTUNE)


# Model Creation
def create_model():
    with strategy.scope():
        # input_layer = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

        # vgg = tf.keras.applications.VGG16(include_top=False)(input_layer)

        # pool = layers.GlobalMaxPooling2D()(vgg)

        # dense1 = layers.Dense(1024, activation='relu')(pool)

        # output = layers.Dense(1, activation='sigmoid')(dense1)

        # model = tf.keras.Model(inputs=input_layer, outputs=output)


        model = tf.keras.Sequential()

        model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

        model.add(layers.RandomRotation(0.3))
        model.add(layers.RandomFlip('horizontal'))

        model.add(layers.Conv2D(26, 3, padding='same', activation='relu'))
        model.add(layers.Conv2D(26, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())

        model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())

        model.add(layers.Flatten())

        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.015, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy'],
        )

        return model

with strategy.scope():
    model = create_model()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

print("Starting Training")


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    batch_size=BATCH_SIZE,
    use_multiprocessing=True,
    callbacks=[early_stopping],    
    workers=32,
)

metrics = model.evaluate(test_dataset)
print(metrics)
model.save(f"ViolenceModel_EvalAcc_{metrics[1]}.h5")