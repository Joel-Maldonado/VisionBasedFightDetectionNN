import pandas as pd
import cv2
import os

DATA_DIR = 'BoundingBoxImages'
LABEL_DIR = './violence_bounding_box_labels.csv'

labels = pd.read_csv(LABEL_DIR)
labels['Path'] = labels['ImageID'].apply(lambda x: os.path.join(DATA_DIR, x))

IMG_SIZE = 256
print(labels)

for index, row in labels.iterrows():
    img = cv2.imread(row['Path'])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    x1, y1, x2, y2 = row[['XMin', 'YMin', 'XMax', 'YMax']]

    x1 = int(x1 * IMG_SIZE)
    y1 = int(y1 * IMG_SIZE)
    x2 = int(x2 * IMG_SIZE)
    y2 = int(y2 * IMG_SIZE)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
