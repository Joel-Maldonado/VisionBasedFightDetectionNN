import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_RECORD_DIR = 'violence_rgb_opt_train.tfrecord'
VAL_RECORD_DIR = 'violence_rgb_opt_val.tfrecord'

VIOLENCE_MODEL_PATH = 'MODEL_SAVE_PATH'
BOX_MODEL_PATH = 'MODEL_SAVE_PATH'

# Constants
SEED = 435
IMG_SIZE = 224
N_FRAMES = 20
BATCH_SIZE = 16
CHANNELS = 5
AUTOTUNE = tf.data.experimental.AUTOTUNE

def parse_tfrecord(example):
  features = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'video': tf.io.FixedLenFeature([], tf.string)
  }
  example = tf.io.parse_single_example(example, features)
  video = tf.io.decode_raw(example['video'], tf.float32)
  video = tf.reshape(video, (N_FRAMES, IMG_SIZE, IMG_SIZE, 5))
  label = tf.cast(example['label'], tf.uint8)

  return video, label

def preprocess(video, label):
    video = video / 255.0
    return video, label

val_dataset = tf.data.TFRecordDataset(VAL_RECORD_DIR)
val_dataset = val_dataset.map(parse_tfrecord)
val_dataset = val_dataset.map(preprocess)

violence_model = tf.keras.models.load_model(VIOLENCE_MODEL_PATH)

labels = []
preds = []
count = 0

for video, label in val_dataset:
  labels.append(label.numpy())

  vid_pred = tf.expand_dims(video, 0)
  pred = tf.round(violence_model(vid_pred, training=False))
  pred = pred[0][0]
  preds.append(pred.numpy())

  count += 1

cm = confusion_matrix(labels, preds)

group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ['{0:0.0f}'.format(x) for x in cm.flatten()]
group_percentages = ['{0:.2f}%'.format(x * 100.0 / np.sum(cm)) for x in cm.flatten()]
group_colors = ['green' if x == 'True Neg' else 'red' for x in group_names]

labels = [f"{group_names[i]}: {group_counts[i]} ({group_percentages[i]})" for i in range(len(group_names))]

labels = np.array(labels).reshape(2,2)

ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

ax.set_title('Confusion Matrix', fontsize=20, y=1.15)
ax.set_xlabel('Predicted Values', fontsize=14)
ax.set_ylabel('Actual Values', fontsize=14)

ax.xaxis.set_ticklabels(['NonViolent', 'Violent'])
ax.yaxis.set_ticklabels(['NonViolent', 'Violent'])

plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')

plt.tight_layout()

plt.show()
