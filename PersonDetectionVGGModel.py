from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import pandas as pd
import tensorflow as tf
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

# Config
AUTOTUNE = tf.data.experimental.AUTOTUNE
SIZE = 224
BATCH_SIZE = 64
DATA_DIR = 'ImageData'

# Load csv
df = pd.read_csv('image_data.csv')
df['ImagePath'] = DATA_DIR + '/' + df['ImageID'].astype(str) + '.jpg'
df['ImagePath'] = df['ImagePath'].astype(str)

DATASET_SIZE = len(df)

# Processing Functions
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [SIZE, SIZE])
    image /= 255.0

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

print(DATASET_SIZE)

# Load into dataset
path_ds = tf.data.Dataset.from_tensor_slices(df['ImagePath'].to_numpy())

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(df[['YMin', 'XMin', 'YMax', 'XMax']].to_numpy(), tf.float16))

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

# ds = image_label_ds.batch(BATCH_SIZE)
ds = image_label_ds.shuffle(1000, seed=SEED)
# ds = image_label_ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

# Train, test, val
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

train_dataset = ds.take(train_size)
test_dataset = ds.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

train_dataset = train_dataset.batch(BATCH_SIZE).cache()
test_dataset = test_dataset.batch(BATCH_SIZE).cache()
val_dataset = val_dataset.batch(BATCH_SIZE).cache()

train_dataset.prefetch(AUTOTUNE)
test_dataset.prefetch(AUTOTUNE)
val_dataset.prefetch(AUTOTUNE)


# learning_rate: 0.0051180353344156115
# optimizer: SGD
# rotation_factor: 0.29167859465591744
# n_conv_layer_groups: 0
# n_dense_layers: 3
# conv_0_1_f: 16
# conv_{i}_1_k: 7
# conv_0_2_f: 66
# max_pooling_{i}?: True
# dropout_0: 0.16116974212466129
# conv_1_1_f: 28
# conv_1_2_f: 110
# dropout_1: 0.46074398321732074
# dense_0_u: 964
# dropout_dense_0: 0.08343701542100612
# dense_u: 2267
# drop_rt: 0.40050642993841573
# lr: 0.03732872467718392
# momentum: 0.907485920388842
# Score: 0.16499446332454681

# Model Creation
def create_model():
    with strategy.scope():
        input_layer = layers.Input(shape=(SIZE, SIZE, 3))

        # aug1 = layers.RandomRotation(0.3)(input_layer)
        # aug2 = layers.RandomFlip('horizontal')(aug1)

        vgg = tf.keras.applications.VGG19(weights='imagenet', include_top=False)(input_layer)

        vgg.trainable = False

        gmp = layers.GlobalMaxPooling2D()(vgg)

        regress1 = layers.Dense(2048, activation='elu')(gmp)
        drop = layers.Dropout(0.5)(regress1)

        regress_output = layers.Dense(4, activation='sigmoid')(drop)


        model = tf.keras.Model(inputs=input_layer, outputs=regress_output)
        
        model.compile(
            loss='mae',
            optimizer= tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['mae', 'mse'],
        )

        return model


early_stopper = tf.keras.callbacks.EarlyStopping('val_mae', patience=5, restore_best_weights=True)

with strategy.scope():
    model = create_model()

history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    epochs=1, 
                    callbacks=[early_stopper], 
                    use_multiprocessing=True, 
                    workers=32,
                    batch_size=BATCH_SIZE)

model.save(f'PersonDetection_mae_temp.h5')

print(test_dataset)
metrics = model.evaluate(test_dataset)

model.save(f'PersonDetection_loss_{metrics[0]}.h5')