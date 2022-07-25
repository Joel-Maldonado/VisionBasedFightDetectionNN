import cv2
import tensorflow as tf

VIDEO_PATH = '/u/jruiz_intern/jruiz/Datasets/fight-detection-surv-dataset-master/fight/fi014.mp4'
# VIDEO_PATH = '/u/jruiz_intern/jruiz/Datasets/fight-detection-surv-dataset-master/noFight/nofi009.mp4'

VIOLENCE_MODEL = tf.keras.models.load_model(
    'Models/Violence_Acc_0.7174999713897705.h5'
)

IMG_SIZE = (224, 224)
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
            frames.append(frame)

    return frames

cap = cv2.VideoCapture(VIDEO_PATH)
frames = get_frames(cap)

for frame in frames:
    new_img = cv2.resize(frame, IMG_SIZE)
    cv2.imshow('frame', new_img)
    cv2.waitKey(2000 // N_FRAMES)