import os
import pickle
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

model_path = 'ViolenceModel_EvalAcc_0.8444444537162781.h5'

video = '/u/jruiz_intern/jruiz/Datasets/RWF-2000/train/Fight/0lHQ2f0d_3.avi'

model = load_model(model_path)

# Constants
MAX_FRAMES = 10
IMG_SIZE = 224


def get_frames(video_path):
    """
    This will return the correct number of frames normalized and resized.
    """
    
    frames = []
    
    vid_reader = cv2.VideoCapture(video_path)
    vid_frames_count = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    amnt_to_skip = max(vid_frames_count // MAX_FRAMES, 1)
    
    for i in range(MAX_FRAMES):
        # Set position of reader
        vid_reader.set(cv2.CAP_PROP_FRAME_COUNT, i * amnt_to_skip)
        
        success, frame = vid_reader.read()
        
        if not success:
            break
        
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        
        normalized_frame = frame_resized / 255.0
        frames.append(normalized_frame)
        
    vid_reader.release()
    return frames

video_frames = tf.expand_dims(tf.constant(get_frames(video)), axis=0)

prediction = int(model.predict(video_frames))


print(prediction)