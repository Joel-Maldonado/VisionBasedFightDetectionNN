import tensorflow as tf
import cv2
import numpy as np
import os


VIDEO = '/u/jruiz_intern/jruiz/Datasets/RWF/val/Fight/6C7PHYI3L9.avi'
# VIDEO = '/u/jruiz_intern/jruiz/Datasets/RWF/train/Fight/4Y23PRE2VD.avi'
# VIDEO = '/u/jruiz_intern/jruiz/Datasets/fight-detection-surv-dataset-master/fight/8RFO63636K.avi'
# VIDEO = '/u/jruiz_intern/jruiz/Downloads/DontDelete/9AB9HPN0N0_CUT.mp4'
# VIDEO = '/u/jruiz_intern/jruiz/Datasets/RWF/train/NonFight/0UHSN1QYG0.avi'
# VIDEO = '/u/jruiz_intern/jruiz/Datasets/RWF/val/NonFight/80N1GRREHM.avi'

VIOLENCE_MODEL_PATH = 'Models/Violence_Acc_0.7649999856948853.h5'
BOX_MODEL_PATH = 'Models/Bounding_Box_GIoU_0.01907881535589695.h5'

# Constants
IMG_SIZE = 224
N_FRAMES = 20

class RandomFlipVideo(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(RandomFlipVideo, self).__init__()

  @tf.function
  def call(self, inputs):
    # if tf.random.uniform(()) > 0.5:
    #   return tf.map_fn(lambda x: tf.image.flip_left_right(x), inputs)
    return inputs
  
class RandomRotationVideo(tf.keras.layers.Layer):
  def __init__(self, max_rotation=0.3, **kwargs):
    super(RandomRotationVideo, self).__init__()
    self.max_rotation = max_rotation

  @tf.function
  def call(self, inputs):
    # random_factor = tf.random.uniform(()) * self.max_rotation * 2 - self.max_rotation
    # return tf.map_fn(lambda x: tfa.image.rotate(x, random_factor), inputs)
    return inputs
    
  def get_config(self):
    config = super().get_config().copy()
    return config

def get_unaltered_frames(cap):
    frames = []
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()
    return frames

def getOpticalFlow(video):
    gray_videos = []
    for img in video:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_videos.append(np.reshape(gray,(IMG_SIZE, IMG_SIZE, 1)))
        
    flows = []
    for i in range(0, len(video)-1):
        flow = cv2.calcOpticalFlowFarneback(gray_videos[i], gray_videos[i+1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        
        # Subtract mean
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])

        # Normalize
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)

        flows.append(flow)

    # Padding last frame
    flows.append(np.zeros((IMG_SIZE, IMG_SIZE, 2)))

    return np.array(flows, dtype=np.float32)


def get_rgb_opt_video(file_path, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(file_path)
    frames = []

    ret, frame = cap.read()
    while ret:
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.reshape(frame, (IMG_SIZE, IMG_SIZE, 3))
        frames.append(frame)
        ret, frame = cap.read()

    frames = np.array(frames)
    cap.release()
            
    flows = getOpticalFlow(frames)
    
    result = np.zeros((len(flows), IMG_SIZE, IMG_SIZE, frames.shape[-1] + flows.shape[-1]))
    result[..., :3] = frames
    result[..., 3:] = flows
    
    return result

cap = cv2.VideoCapture(VIDEO)
unaltered_frames = get_unaltered_frames(cap)

fps = cap.get(cv2.CAP_PROP_FPS)
h, w = unaltered_frames[0].shape[:2]

cap.release()


extracted_full = get_rgb_opt_video(VIDEO) / 255.0
extracted_cut = np.array([extracted_full[int(i)] for i in np.linspace(0, len(extracted_full)-1, N_FRAMES)])

violent_model = tf.keras.models.load_model(VIOLENCE_MODEL_PATH, custom_objects={'RandomFlipVideo': RandomFlipVideo, 'RandomRotationVideo': RandomRotationVideo})
box_model = tf.keras.models.load_model(BOX_MODEL_PATH, compile=False)


violent_pred = violent_model.predict(tf.expand_dims(extracted_cut, axis=0))[0, 0]

predicted_frames = []
for unaltered_frame, extracted_frame in zip(unaltered_frames, extracted_full):
    formatted = cv2.cvtColor(tf.cast(extracted_frame[..., :3] * 255.0, np.uint8).numpy(), cv2.COLOR_RGB2BGR)
    gray = tf.image.rgb_to_grayscale(extracted_frame[..., :3])
    box_pred = box_model.predict(tf.expand_dims(gray, axis=0))[0]

    if violent_pred > 0.5:
        y1, x1, y2, x2 = box_pred

        y1 = int(y1 * h)
        x1 = int(x1 * w)
        y2 = int(y2 * h)
        x2 = int(x2 * w)


        cv2.rectangle(unaltered_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    violence_text = f"Violence: {(violent_pred * 100):.2f}%"
    cv2.putText(unaltered_frame, violence_text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    predicted_frames.append(unaltered_frame)


out_name = f"/u/jruiz_intern/jruiz/Downloads/DontDelete/PredictDump/{os.path.basename(VIDEO).split('.')[0]}_LABELED.avi"

out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
for extracted_frame in predicted_frames:
    out.write(extracted_frame)
out.release()
