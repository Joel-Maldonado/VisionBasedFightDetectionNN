import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
import os

IMG_SIZE = 256
MAX_FRAME_PREDICTIONS = 9999

def get_frames(cap):
    frames = []

    ret, img = cap.read()
    while ret:
        frames.append(img)
        ret, img = cap.read()
    
    print(f"Found {len(frames)} frames.")
    return frames

def get_rolling_average(arr, window_size=10):
    numbers_series = pd.Series(arr)

    windows = numbers_series.rolling(window_size)

    moving_averages = windows.mean()

    moving_averages_list = moving_averages.tolist()

    final_list = moving_averages_list[window_size - 1:]
    
    return final_list


model = tf.keras.models.load_model('./ViolenceModel_EvalAcc_0.9399999976158142.h5')

vid_path = '/u/jruiz_intern/Desktop/jruiz/Datasets/RWF-2000/val/Fight/nuf-d5GugL0_7.avi'
vid_name = os.path.basename(vid_path).split('.')[0]

cap = cv2.VideoCapture(vid_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width,frame_height)
fps = round(cap.get(cv2.CAP_PROP_FPS))

print(frame_size, fps)

frames = get_frames(cap)

preds = []

for count, index in enumerate(np.linspace(0, len(frames)-1, min(len(frames), MAX_FRAME_PREDICTIONS))):
    img = frames[int(index)]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_predict = tf.expand_dims(img / 255.0, 0)

    pred = model.predict(img_predict)
    preds.append(pred)

    print(f"Predicted frames {count+1}/{min(len(frames), MAX_FRAME_PREDICTIONS)}")

average = np.average(preds)
rolling_average = get_rolling_average(preds)

i = 0
rate = len(rolling_average) / len(frames)
for frame in frames:
    i += rate

    binary = round(rolling_average[round(i)-1])
    pred_str = 'Violent' if binary == 1 else 'Non-Violent'
    cv2.putText(frame, pred_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, str(average), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

out = cv2.VideoWriter(f'{vid_name}_LABELED.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))

print("Writing File")
for frame in frames:
    out.write(frame)

out.release()
cv2.destroyAllWindows() 
print("Done.")