import tensorflow as tf
import cv2
import numpy as np
import os

# VIDEO_PATH = '/u/jruiz_intern/jruiz/Datasets/fight-detection-surv-dataset-master/fight/fi014.mp4'
VIDEO_PATH = '/u/jruiz_intern/jruiz/Datasets/fight-detection-surv-dataset-master/noFight/nofi026.mp4'

VIOLENCE_MODEL_PATH = 'Models/violence_model_acc_96.h5'



IMG_SIZE = 224
N_FRAMES = 20

def get_frames(cap):
    frames = []
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    skip_rate = int(total_frames / N_FRAMES)
    for i in range(N_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_rate)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)

    return frames

cap = cv2.VideoCapture(VIDEO_PATH)
frames = get_frames(cap)

fps = cap.get(cv2.CAP_PROP_FPS)
h, w = frames[0].shape[:2]

cap.release()

frames = tf.cast(frames, tf.float32)
frames = tf.reshape(frames, (1, N_FRAMES, IMG_SIZE, IMG_SIZE, 1))
frames = frames / 255.0

prediction = VIOLENCE_MODEL.predict(frames)
print(f'Prediction: {prediction[0][0]}')
print('Violent' if round(prediction[0][0]) else 'Non-Violent')