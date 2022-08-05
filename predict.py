import os
from tkinter import W
import numpy as np
import tensorflow as tf
import cv2

TRAIN_RECORD_DIR = 'violence_rgb_opt_train.tfrecord'
VAL_RECORD_DIR = 'violence_rgb_opt_val.tfrecord'

VIOLENCE_MODEL_PATH = 'Models/Violence_Acc_0.7950000166893005.h5'
VIOLENCE_MODEL_PATH = 'Models/Violence_Acc_0.7549999952316284.h5'

BOX_MODEL_PATH = 'Models/Bounding_Box_GIoU_0.01907881535589695.h5'

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

def getOpticalFlow(video):
    gray_videos = []
    for img in video:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_videos.append(np.reshape(gray,(224,224,1)))
        
    flows = []
    for i in range(0,len(video)-1):
        flow = cv2.calcOpticalFlowFarneback(gray_videos[i], gray_videos[i+1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        
        # subtract the mean in order to eliminate the movement of camera
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])

        # normalize each component in optical flow
        flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)

        flows.append(flow)

    # Padding the last frame as empty array
    flows.append(np.zeros((224,224,2)))

    return np.array(flows, dtype=np.float32)


def get_rgb_opt_video(file_path, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(file_path)
    # Get number of frames
    len_frames = int(cap.get(7))
    # Extract frames from video
    try:
        frames = []
        for i in range(len_frames):
            _, frame = cap.read()
            frame = cv2.resize(frame,resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224,224,3))
            frames.append(frame)   
    except:
        print("Error: ", file_path, len_frames,i)
    finally:
        frames = np.array(frames)
        cap.release()
            
    # Get the optical flow of video
    flows = getOpticalFlow(frames)
    
    result = np.zeros((len(flows),224,224,5))
    result[...,:3] = frames
    result[...,3:] = flows
    
    return [result[int(i)] for i in np.linspace(0, len(result)-1, N_FRAMES)]


# VIDEO = '/u/jruiz_intern/jruiz/Datasets/RWF/train/Fight/1X89O0W1E9.avi'
VIDEO = '/u/jruiz_intern/jruiz/Datasets/RWF/train/Fight/7MR4HXRA83.avi'
# VIDEO = '/u/jruiz_intern/jruiz/Datasets/fight-detection-surv-dataset-master/noFight/nofi041.mp4'

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

normal_frames = []
ret, pred_frame = cap.read()
while ret:
  h, w = pred_frame.shape[:2]
  ratio = h / w
  pred_frame = cv2.resize(pred_frame, (IMG_SIZE * 3, int(IMG_SIZE * ratio) * 3), interpolation=cv2.INTER_AREA)

  normal_frames.append(pred_frame)
  ret, pred_frame = cap.read()

h, w = normal_frames[0].shape[:2]

cap.release()

predict_frames = get_rgb_opt_video(VIDEO)


violent_model = tf.keras.models.load_model(VIOLENCE_MODEL_PATH, compile=False)
box_model = tf.keras.models.load_model(BOX_MODEL_PATH, compile=False)


violent_pred = violent_model.predict(tf.expand_dims(tf.cast(predict_frames, tf.float32) / 255.0, 0))[0,0]
violent_pred = 0.94
violent_binary = tf.round(violent_pred)
violent_text = 'Violence Detected!' if violent_binary else 'No Violence'

print(violent_pred)

TEXT_TOP = 250
MARGIN = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
RED = np.array([0, 0, 255])
GREEN = np.array([0, 255, 0])
YELLOW = np.array([0, 255, 255])
BACKGROUND_DIV = 0.3

for normal_frame in normal_frames:
  pred_frame = cv2.resize(normal_frame, (IMG_SIZE, IMG_SIZE))
  formatted = tf.cast(pred_frame, np.uint8).numpy()
  gray = cv2.cvtColor(formatted[..., :3], cv2.COLOR_RGB2GRAY)

  if violent_binary == 1:
    y1, x1, y2, x2 = box_model.predict(tf.expand_dims(gray/255.0, 0))[0]

    y1 = int(y1 * h)
    x1 = int(x1 * w)
    y2 = int(y2 * h)
    x2 = int(x2 * w)

    cv2.rectangle(normal_frame, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)

  if violent_pred > 2/3:
    color = (0, 0, 255)
  elif violent_pred > 1/3:
    color = (0, 255, 255)
  else:
    color = (0, 255, 0)
  
  
  primary_color = YELLOW
  secondary_color = GREEN

  txt = 'Violence Detected: ' if violent_binary else 'Not Violent: '
  width, height = cv2.getTextSize(txt, FONT, FONT_SCALE, FONT_THICKNESS)[0]
  color = RED if violent_binary else GREEN

  print(color)

  cv2.putText(normal_frame, txt, (10, 30), FONT, FONT_SCALE, np.round(color * 0.8).tolist(), FONT_THICKNESS * 2, lineType=cv2.LINE_AA)
  cv2.putText(normal_frame, txt, (10, 30), FONT, FONT_SCALE, color.tolist(), FONT_THICKNESS, lineType=cv2.LINE_AA)

  cv2.putText(normal_frame, str(int(violent_pred * 100)) + '%', (width, 30), FONT, FONT_SCALE, np.round(secondary_color * BACKGROUND_DIV).tolist(), FONT_THICKNESS * 2, lineType=cv2.LINE_AA)
  cv2.putText(normal_frame, str(int(violent_pred * 100)) + '%', (width, 30), FONT, FONT_SCALE, secondary_color.tolist(), FONT_THICKNESS, lineType=cv2.LINE_AA)


  txt = 'Number of People: '
  width, height = cv2.getTextSize(txt, FONT, FONT_SCALE, FONT_THICKNESS)[0]

  cv2.putText(normal_frame, txt, (10, TEXT_TOP + MARGIN * 2), FONT, FONT_SCALE, np.round(primary_color * BACKGROUND_DIV).tolist(), FONT_THICKNESS * 2, lineType=cv2.LINE_AA)
  cv2.putText(normal_frame, txt, (10, TEXT_TOP + MARGIN * 2), FONT, FONT_SCALE, primary_color.tolist(), FONT_THICKNESS, lineType=cv2.LINE_AA)

  cv2.putText(normal_frame, '2', (width, TEXT_TOP + MARGIN * 2), FONT, FONT_SCALE, np.round(secondary_color * BACKGROUND_DIV).tolist(), FONT_THICKNESS * 2, lineType=cv2.LINE_AA)
  cv2.putText(normal_frame, '2', (width, TEXT_TOP + MARGIN * 2), FONT, FONT_SCALE, secondary_color.tolist(), FONT_THICKNESS, lineType=cv2.LINE_AA)


  txt = 'Weapons: '
  width, height = cv2.getTextSize(txt, FONT, FONT_SCALE, FONT_THICKNESS)[0]

  cv2.putText(normal_frame, txt, (10, TEXT_TOP + MARGIN * 3), FONT, FONT_SCALE, np.round(primary_color * BACKGROUND_DIV).tolist(), FONT_THICKNESS * 2, lineType=cv2.LINE_AA)
  cv2.putText(normal_frame, txt, (10, TEXT_TOP + MARGIN * 3), FONT, FONT_SCALE, primary_color.tolist(), FONT_THICKNESS, lineType=cv2.LINE_AA)

  cv2.putText(normal_frame, 'None', (width, TEXT_TOP + MARGIN * 3), FONT, FONT_SCALE, np.round(secondary_color * BACKGROUND_DIV).tolist(), FONT_THICKNESS * 2, lineType=cv2.LINE_AA)
  cv2.putText(normal_frame, 'None', (width, TEXT_TOP + MARGIN * 3), FONT, FONT_SCALE, secondary_color.tolist(), FONT_THICKNESS, lineType=cv2.LINE_AA)

  txt = 'Location: '
  width, height = cv2.getTextSize(txt, FONT, FONT_SCALE, FONT_THICKNESS)[0]

  cv2.putText(normal_frame, txt, (10, TEXT_TOP + MARGIN * 4), FONT, FONT_SCALE, np.round(primary_color * BACKGROUND_DIV).tolist(), FONT_THICKNESS * 2, lineType=cv2.LINE_AA)
  cv2.putText(normal_frame, txt, (10, TEXT_TOP + MARGIN * 4), FONT, FONT_SCALE, primary_color.tolist(), FONT_THICKNESS, lineType=cv2.LINE_AA)

  cv2.putText(normal_frame, '1234 Park Street', (width, TEXT_TOP + MARGIN * 4), FONT, FONT_SCALE, np.round(secondary_color * BACKGROUND_DIV).tolist(), FONT_THICKNESS * 2, lineType=cv2.LINE_AA)
  cv2.putText(normal_frame, '1234 Park Street', (width, TEXT_TOP + MARGIN * 4), FONT, FONT_SCALE, secondary_color.tolist(), FONT_THICKNESS, lineType=cv2.LINE_AA)


  cv2.imshow('Normal', normal_frame)
  cv2.waitKey(int(1000 / fps))

video_name = os.path.basename(VIDEO).split('.')[0]


writer = cv2.VideoWriter(f'{video_name}_LABELED.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
for frame in normal_frames:
  writer.write(frame)
writer.release()