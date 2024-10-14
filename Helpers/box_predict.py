import os
import tensorflow as tf
import cv2
import numpy as np
import tensorflow_addons as tfa
import pandas as pd


CHANNELS = 1


@tf.function
def GIoU_MSE(y_true, y_pred):
    giou = tfa.losses.giou_loss(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    return giou + mse


def get_rolling_average(arr, window_size=3):
    numbers_series = pd.Series(arr)

    windows = numbers_series.rolling(window_size)

    moving_averages = windows.mean()

    moving_averages_list = moving_averages.tolist()

    final_list = moving_averages_list[window_size - 1 :]

    return final_list


def get_frames(cap):
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


@tf.function
def preprocess_image(frame_cv):
    img_cast = tf.cast(frame_cv, tf.float32)
    img_tensor = tf.convert_to_tensor(img_cast, dtype=tf.float32)

    img_predict = tf.image.rgb_to_grayscale(img_tensor)
    img_predict = tf.image.resize(img_predict, [IMG_SIZE, IMG_SIZE]) / 255.0

    return tf.expand_dims(img_predict, axis=0)


BOX_MODEL = tf.keras.models.load_model(
    "Models/Bounding_Box_GIoU_0.026832809671759605.h5",
    custom_objects={"GIoU_MSE": GIoU_MSE},
)

VIOLENCE_MODEL = tf.keras.models.load_model("Models/Violence_Acc_0.9865145087242126.h5")


VIOLENCE_DIR = "~/RWF-2000/val/Fight"
NON_VIOLENCE_DIR = "~/RWF-2000/val/NonFight"

NUM_VIDS_PER_CATEGORY = 5

for category_num, category in enumerate([VIOLENCE_DIR, NON_VIOLENCE_DIR]):
    videos = os.listdir(category)
    np.random.shuffle(videos)

    IMG_SIZE = 256
    for vid_num, video in enumerate(videos[:NUM_VIDS_PER_CATEGORY]):

        vid_path = os.path.join(category, video)
        cap = cv2.VideoCapture(vid_path)

        frames = get_frames(cap)

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        h, w, _ = frames[0].shape
        vid_name = os.path.basename(vid_path).split(".")[0]

        box_preds = []
        violence_preds = []

        processed_frames = []
        for i, frame in enumerate(frames):
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predict_img = preprocess_image(frame)

            y1, x1, y2, x2 = BOX_MODEL.predict(predict_img, verbose=0)[0]
            violence = VIOLENCE_MODEL.predict(predict_img, verbose=0)[0][0]

            y1 = int(y1 * h)
            x1 = int(x1 * w)
            y2 = int(y2 * h)
            x2 = int(x2 * w)

            violence_preds.append(violence)
            box_preds.append([y1, x1, y2, x2])

        roll_violence_preds = get_rolling_average(violence_preds, window_size=10)

        # apply the rolling average to each frame in frames
        for i, frame in enumerate(frames):
            roll_violence_i = int((i + 1) / len(frames) * len(roll_violence_preds)) - 1

            y1, x1, y2, x2 = box_preds[i]
            violence = roll_violence_preds[roll_violence_i]

            img = tf.cast(frame, tf.uint8).numpy()

            if violence > 0.5:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.putText(
                img,
                f"Violent: {(violence * 100):.2f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            processed_frames.append(cv2.resize(img, (w, h)))

        out = cv2.VideoWriter(
            f"Predictions/{vid_name}_LABELED.avi",
            cv2.VideoWriter_fourcc(*"DIVX"),
            fps,
            (w, h),
        )

        for frame in processed_frames:
            out.write(frame)

        out.release()
        cv2.destroyAllWindows()

        print(
            f"Finished with Video: {vid_num+1}/{NUM_VIDS_PER_CATEGORY} | Category: {category_num+1}/{2}"
        )
