import tensorflow as tf
import cv2
import numpy as np
import os
import tensorflow_addons as tfa
from multiprocessing import Pool
from tqdm import tqdm

VIDEO_PATH = '/u/jruiz_intern/jruiz/Datasets/fight-detection-surv-dataset-master/fight/fi104.mp4'
# VIDEO_PATH = '/u/jruiz_intern/jruiz/Datasets/fight-detection-surv-dataset-master/noFight/nofi0.mp4'
DIR_PATH = '/u/jruiz_intern/jruiz/Datasets/fight-detection-surv-dataset-master/fight'

# VIOLENCE_MODEL_PATH = 'Models/violence_model_acc_96.h5'
VIOLENCE_MODEL_PATH = 'Models/Violence_Acc_0.7574999928474426.h5'

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

model = tf.keras.models.load_model(VIOLENCE_MODEL_PATH, custom_objects={'RandomFlipVideo': RandomFlipVideo, 'RandomRotationVideo': RandomRotationVideo})


IMG_SIZE = 224
N_FRAMES = 20

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
    paths = [os.path.join(directory, file) for file in os.listdir(directory)]
    pool = Pool()
    
    vids = []
    for vid in tqdm(pool.imap_unordered(get_rgb_opt_video, paths), total=len(paths)):
        vids.append(vid)
    
    return vids


correct = 0
total = 0
for video in os.listdir(DIR_PATH):
  path = os.path.join(DIR_PATH, video)
  vid = get_rgb_opt_video(path)

  pred = model.predict(tf.expand_dims(vid, 0))[0][0]
  
  if pred > 0.5:
    correct += 1
    total += 1
  else:
    total +=  1

  print(f"Correct: {correct}")
  print(f"Total: {total}")
  print(f"Accuracy: {correct/total * 100}%")


# frames = get_rgb_opt_video(VIDEO_PATH)

# print(tf.cast(frames, tf.float32).shape)
# frames = tf.cast(frames, tf.float32)
# print(frames)


# pred = model.predict(tf.expand_dims(frames, 0)/255.0)[0][0]
# pred = model.predict(tf.expand_dims(frames, 0))[0][0]
# print(pred)

# print({'Violent' if pred > 0.5 else 'NonViolent'})

# print(tf.expand_dims(frames, 0)/255.0)

# print(f"Prediction: {pred}")
# # cv2.waitKey(0)

# for frame in frames:
#     formatted = tf.cast(frame, np.uint8).numpy()
#     rgb = formatted[...,:3]
#     bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#     flow = np.hstack((formatted[...,3], formatted[...,4]))

#     bgr_title = f"RGB | {'Violent' if pred > 0.5 else 'NonViolent'}"
#     cv2.imshow(bgr_title, bgr)
#     cv2.moveWindow(bgr_title, 0, 0)
    
#     flow_title = f"Optical Flow | {'Violent' if pred > 0.5 else 'NonViolent'}"
#     cv2.imshow(flow_title, flow)
#     cv2.moveWindow(flow_title, IMG_SIZE*2, 0)

#     if cv2.waitKey(5000 // N_FRAMES) == ord('q'):
#         print("Quitting")
#         break
