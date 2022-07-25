# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv('./layer_data2.csv')

# sns.pairplot(df)

# plt.show()

import cv2
import numpy as np
import tensorflow as tf

def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    # if tf.random.uniform(()) > 0.5:
    if True:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


# 1,0.3638888889,0.225,0.7152777778,0.61875,rLkkX1Fr8C2QPBDy.avi_frame9.jpgq

im = tf.io.read_file('ViolenceImg/rLkkX1Fr8C2QPBDy.avi_frame9.jpg')
im = tf.image.decode_image(im, 3)

box = tf.constant([0.3638888889,0.225,0.7152777778,0.61875])


im, box = random_flip_horizontal(im, tf.expand_dims(box, axis=0))
print(im, box)

img = tf.cast(im, np.uint8).numpy()

box = box[0]

x1 = int(box[0] * im.shape[1])
y1 = int(box[1] * im.shape[0])
x2 = int(box[2] * im.shape[1])
y2 = int(box[3] * im.shape[0])

cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('im', img)
cv2.waitKey(0)
