import cv2
import tensorflow as tf
import numpy as np

N_FRAMES = 30
IMG_SIZE = 224


def parse_tfrecord(example):
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'video': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, features)
    # video = tf.io.decode_raw(example['video'], tf.uint8)
    video = tf.io.decode_raw(example['video'], tf.float32)
    video = tf.reshape(video, (N_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    label = tf.cast(example['label'], tf.uint8)
    return video, label

RECORD_PATH = 'test.tfrecord'

data = tf.data.TFRecordDataset(RECORD_PATH)
data = data.map(parse_tfrecord)

for video, label in data.take(1):
  # print(video[..., :3])
  # print(video)
  vid = tf.cast(video, np.uint8)
  for img in vid:
    print(img)
    img = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
    cv2.imshow('video', img)
    cv2.waitKey(0)
#   print(video.shape)
#   print(video[0])
# for vid in data.take(5):
#   print(vid)


# def getOpticalFlow(video):
#     # Grayscale all frames in video
#     gray_video = []
#     for frame in video:
#         gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         gray_video.append(np.reshape(gray,(IMG_SIZE, IMG_SIZE, 1)))

#     # Calculate flows
#     flows = []
#     for i in range(0,len(video)-1):
#         flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i+1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

#         # Subtract the mean in order to eliminate the movement of camera
#         flow[..., 0] -= np.mean(flow[..., 0])
#         flow[..., 1] -= np.mean(flow[..., 1])

#         # Normalize each component in optical flow
#         flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
#         flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)

#         flows.append(flow)
        
#     # Padding the last frame as empty array
#     flows.append(np.zeros((224,224,2)))
      
#     return np.array(flows, dtype=np.float32)

# def get_rgb_opt_video(file_path, resize=(IMG_SIZE, IMG_SIZE)):
#     cap = cv2.VideoCapture(file_path)

#     frames = []
    
#     ret, frame = cap.read()
#     while ret:
#         frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = np.reshape(frame, (IMG_SIZE, IMG_SIZE, 3))
#         frames.append(frame)
#         ret, frame = cap.read()

#     frames = np.array(frames)
#     cap.release()
            
#     flows = getOpticalFlow(frames)
    
#     # Empty array of shape (FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS)
#     combined_vid = np.zeros((frames.shape[0], 224, 224, frames.shape[-1] + flows.shape[-1]))

#     # Fill empty array with combined frames & flows
#     combined_vid[...,:3] = frames
#     combined_vid[...,3:] = flows

#     # Cut video to N_FRAMES
#     # combined_vid = combined_vid[::combined_vid.shape[0]//N_FRAMES]
    
#     return frames[::frames.shape[0]//N_FRAMES]


# vid = get_rgb_opt_video('/u/jruiz_intern/jruiz/Datasets/RWF/train/Fight/0M54F9C21Z.avi')

# print(vid.shape)

# for img in vid:
#   print(img)
#   img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#   cv2.imshow('video', img)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()