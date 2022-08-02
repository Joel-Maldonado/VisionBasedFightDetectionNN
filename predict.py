import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import tensorflow_addons as tfa
from tqdm import tqdm
from multiprocessing import Pool

TRAIN_RECORD_DIR = 'violence_rgb_opt_train.tfrecord'
VAL_RECORD_DIR = 'violence_rgb_opt_val.tfrecord'

VIOLENCE_MODEL_PATH = 'Checkpoints/violence_model_09-50-AM.h5'
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

def random_flip(video, label):
  if tf.random.uniform(()) > 0.5:
    return tf.map_fn(lambda x: tf.image.flip_left_right(x), video), label
  return video, label

def random_rotation(video, label):
  random_factor = tf.random.uniform(()) * 0.3 * 2 - 0.3
  return tf.map_fn(lambda x: tfa.image.rotate(x, random_factor), video), label

def random_reduce_quality(video, label):
  factor = tf.random.uniform(()) * 2.5
  return tf.map_fn(
    lambda x: tf.image.resize(tf.image.resize(x, (int(IMG_SIZE / factor), int(IMG_SIZE / factor))), (IMG_SIZE, IMG_SIZE)), video), label


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


def get_dir_vids(directory):
    paths = [os.path.join(directory, file) for file in os.listdir(directory)][:2]
    pool = Pool()
    
    vids = []
    for vid in tqdm(pool.imap_unordered(get_rgb_opt_video, paths), total=len(paths)):
        vids.append(vid)
    
    return vids

print("Getting violent videos...")
v_videos = get_dir_vids('/u/jruiz_intern/jruiz/Datasets/fight-detection-surv-dataset-master/fight')
v_labels = [1] * len(v_videos)


v_data = list(zip(v_videos, v_labels))


print("Getting nonviolent videos...")
nv_videos = get_dir_vids('/u/jruiz_intern/jruiz/Datasets/fight-detection-surv-dataset-master/noFight')
nv_labels = [0] * len(nv_videos)

nv_data = list(zip(nv_videos, nv_labels))


data = v_data + nv_data
print(len(data))
np.random.shuffle(data)

# def random_blur(video, label):
#   random_factor = tf.random.uniform(()) * 10 + 1
#   return tf.map_fn(lambda x: tfa.image.gaussian_filter2d(x, (random_factor, random_factor)), video), label


# val_dataset = tf.data.TFRecordDataset(VAL_RECORD_DIR)
# val_dataset = val_dataset.map(parse_tfrecord)
# val_dataset = val_dataset.map(preprocess)
# val_dataset = val_dataset.map(random_flip)
# val_dataset = val_dataset.map(random_rotation)
# val_dataset = val_dataset.map(random_reduce_quality)

# violence_model = tf.keras.models.load_model(VIOLENCE_MODEL_PATH)
# box_model = tf.keras.models.load_model(BOX_MODEL_PATH, compile=False)

# for video, label in val_dataset.skip(10).take(10):
#   violent = violence_model.predict(tf.expand_dims(video, 0))
#   violent = round(violent[0][0])

#   for img in video:
#     formatted = tf.cast(img * 255, np.uint8).numpy()

#     gray = tf.image.rgb_to_grayscale(img[..., :3])


#     rgb = formatted[:, :, :3]
#     bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#     opt = np.hstack([formatted[:, :, 3], formatted[:, :, 4]])

#     if violent == 1:
#       y1, x1, y2, x2 = box_model.predict(tf.expand_dims(gray, 0))[0]

#       print(y1, x1, y2, x2)

#       y1 = int(y1 * IMG_SIZE)
#       x1 = int(x1 * IMG_SIZE)
#       y2 = int(y2 * IMG_SIZE)
#       x2 = int(x2 * IMG_SIZE)

#       cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
#     cv2.putText(bgr, str(violent), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow('rgb', bgr)
#     cv2.moveWindow('rgb', 0, 0)
#     cv2.imshow('opt', opt)
#     cv2.moveWindow('opt', IMG_SIZE * 2, 0)

#     key = cv2.waitKey(5000//N_FRAMES)
#     if key == ord('q'):
#       sys.exit()


# model = tf.keras.models.load_model(MODEL_PATH)


# correct = 0
# total = 0

# print("-----------------------------------------------------")

# for video, label in val_dataset.take(50):
#   if round(violence_model.predict(tf.expand_dims(video, 0), verbose=0)[0][0]) == label:
#     correct += 1
#   total += 1

#   print(f"Accuracy: {correct/total} | {correct}/{total}")
#   print("-----------------------------------------------------")

# print(f"Final Accuracy: {correct/total} | {correct}/{total}")


class RandomFlipVideo(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(RandomFlipVideo, self).__init__()

  @tf.function
  def call(self, inputs):
    if tf.random.uniform(()) > 0.5:
      return tf.map_fn(lambda x: tf.image.flip_left_right(x), inputs)
    return inputs
  
class RandomRotationVideo(tf.keras.layers.Layer):
  def __init__(self, max_rotation=0.3, **kwargs):
    super(RandomRotationVideo, self).__init__()
    self.max_rotation = max_rotation

  @tf.function
  def call(self, inputs):
    random_factor = tf.random.uniform(()) * self.max_rotation * 2 - self.max_rotation
    return tf.map_fn(lambda x: tfa.image.rotate(x, random_factor), inputs)
    
  def get_config(self):
    config = super().get_config().copy()
    return config



# def get_prediction(model, data):
#   video, label = data
#   return round(model.predict(tf.expand_dims(video, 0), verbose=0)[0][0]) == label

# results = { }

for model_name in os.listdir('Models'):
  if not 'Violence_Acc_' in model_name:
    continue
  print(f"Loading model {model_name}")
  
#   print(f"Loading model: {model_name}")

  path = os.path.join('Models', model_name)
  try:
    model = tf.keras.models.load_model(path)
  except Exception:
    model = tf.keras.models.load_model(path, compile=True, custom_objects={'RandomFlipVideo': RandomFlipVideo, 'RandomRotationVideo': RandomRotationVideo})

  print(model.evaluate(data))
#   correct = 0
#   total = 0

#   print("-----------------------------------------------------")

#   # for video, label in data:
#   #   if round(model.predict(tf.expand_dims(video, 0), verbose=0)[0][0]) == label:
#   #     correct += 1
#   #   total += 1
    

  
#   pool = Pool()
#   for result in pool.imap_unordered(get_prediction, data):
#     if result:
#       correct += 1
#     total += 1
#     print(f"Accuracy: {correct/total} | {correct}/{total}")
#     print("-----------------------------------------------------")

#   print(f"Final Accuracy: {correct/total} | {correct}/{total}")

#   results[model_name] = correct/total

#   print(results)


# print(f"Best Model: {max(results, key=results.get)}")
# print(f"Best Model Accuracy: {max(results.values())}")