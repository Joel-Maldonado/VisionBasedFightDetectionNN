import cv2
from matplotlib.pyplot import step
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import tensorflow_addons as tfa
import tensorflow_io as tfio
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
IMG_SIZE = 224
N_FRAMES = 20
BATCH_SIZE = 1
CHANNELS = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE

V_DIR = '/u/jruiz_intern/jruiz/Datasets/RWF/val/Fight'
NV_DIR = '/u/jruiz_intern/jruiz/Datasets/RWF/val/NonFight'

def get_frames(vid_path):
    cap = cv2.VideoCapture(vid_path)
    frames = []
    step_size = cap.get(cv2.CAP_PROP_FRAME_COUNT) // N_FRAMES
    for i in range(N_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * step_size))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
        else:
            break
    cap.release()

    frames_tensor = tf.cast(frames, tf.float32)
    return tf.reshape(frames_tensor, (N_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS))


v_videos = []
for i, video in enumerate(os.listdir(V_DIR)):
    path = os.path.join(V_DIR, video)
    frames = get_frames(path)
    v_videos.append(frames)
    print(f'Violence: {i}/{len(os.listdir(V_DIR))}')


nv_videos = []
for i, video in enumerate(os.listdir(NV_DIR)):
    path = os.path.join(NV_DIR, video)
    frames = get_frames(path)
    nv_videos.append(frames)
    print(f'Non_Violence: {i}/{len(os.listdir(V_DIR))}')


v_videos = tf.convert_to_tensor(v_videos)
v_labels = [1] * v_videos.shape[0]

nv_videos = tf.convert_to_tensor(nv_videos)
nv_labels = [0] * nv_videos.shape[0]

videos = tf.concat([v_videos, nv_videos], axis=0)
labels = v_labels + nv_labels

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

with tf.io.TFRecordWriter('violence_video_val.tfrecord') as tfrecord:
    for i, (frame, label) in enumerate(zip(videos, labels)):
        example = tf.train.Example(features=tf.train.Features(feature={
            'feature': wrap_bytes(frame.numpy().tobytes()),
            'label': wrap_int64(label)
        }))
        tfrecord.write(example.SerializeToString())

        print(f"{i}/{len(videos)}")

    