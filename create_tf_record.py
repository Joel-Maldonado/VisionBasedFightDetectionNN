import numpy as np
import cv2
import tensorflow as tf
import os
from tqdm import tqdm
from multiprocessing import Pool

N_FRAMES = 20
IMG_SIZE = 224
CHANNELS = 5

DIR = '/u/jruiz_intern/jruiz/Datasets/RWF/train'

V_DIR = os.path.join(DIR, 'Fight')
NV_DIR = os.path.join(DIR, 'NonFight')

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

print("Getting violent videos...")
v_videos = get_dir_vids(V_DIR)
v_labels = [1] * len(v_videos)


v_data = list(zip(v_videos, v_labels))


print("Getting nonviolent videos...")
nv_videos = get_dir_vids(NV_DIR)
nv_labels = [0] * len(nv_videos)

nv_data = list(zip(nv_videos, nv_labels))


data = v_data + nv_data
print(len(data))
np.random.shuffle(data)


print("Creating TFRecord...")

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

with tf.io.TFRecordWriter('violence_rgb_opt_train.tfrecord') as tfrecord:

    for video, label in tqdm(data):

        vid = tf.cast(video, tf.float32)
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'video': wrap_bytes(vid.numpy().tobytes()),
            'label': wrap_int64(label)
        }))
        tfrecord.write(example.SerializeToString())

    # print("Writing NONVIOLENT videos to TFRecord...")
    # print("Writing VIOLENT videos to TFRecord...")
    # for i, (video, label) in tqdm(enumerate(zip(v_videos, v_labels)), total=len(v_videos)):

    #     vid = tf.cast(video, tf.float32)

    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         'video': wrap_bytes(vid.numpy().tobytes()),
    #         'label': wrap_int64(label)
    #     }))
    #     tfrecord.write(example.SerializeToString())

    # print("Writing NONVIOLENT videos to TFRecord...")
    # for i, (video, label) in tqdm(enumerate(zip(nv_videos, nv_labels)), total=len(nv_videos)):

    #     vid = tf.cast(video, tf.float32)

    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         'video': wrap_bytes(vid.numpy().tobytes()),
    #         'label': wrap_int64(label)
    #     }))
    #     tfrecord.write(example.SerializeToString())