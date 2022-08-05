import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


accs = [77.0, 82.75, 87.25, 80.5]
labels = ['ConvLSTM', 'C3D', 'Fusion (P3D)', 'My Model']
# df = pd.DataFrame({'Accuracy': [their_acc, my_acc], 'Parameters': [their_params, my_params]})

# df.T.plot.bar()
# plt.show()

bp = sns.barplot(x=labels, y=accs, palette='coolwarm')
bp.set_ylim(50, 100)
bp.set_xlabel('Model', fontsize=14)
bp.set_ylabel('Accuracy', fontsize=14)
bp.set_title('Violence Accuracy Comparison on RWF-2000', fontsize=18, y=1.15)
bp.set_yticklabels([f'{x}%' for x in bp.get_yticks()], size=14)
bp.set_xticklabels(labels, fontsize=12)
plt.tight_layout()
plt.show()


# train_acc = pd.read_csv(
#     '/u/jruiz_intern/jruiz/Downloads/DontDelete/ViolenceModel/ViolenceModel79/run-train-tag-epoch_accuracy.csv'
# )

# val_acc = pd.read_csv(
#     '/u/jruiz_intern/jruiz/Downloads/DontDelete/ViolenceModel/ViolenceModel79/run-validation-tag-epoch_accuracy.csv',
# )

# train_loss = pd.read_csv(
#     '/u/jruiz_intern/jruiz/Downloads/DontDelete/BoundingBoxModel/run-train-tag-epoch_loss.csv'
# )

# val_loss = pd.read_csv(
#     '/u/jruiz_intern/jruiz/Downloads/DontDelete/BoundingBoxModel/run-validation-tag-epoch_loss.csv',
# )

# plt.plot(train_loss['Step'], train_loss['Value'], label='Train')
# plt.plot(val_loss['Step'], val_loss['Value'], label='Validation')
# plt.title("Loss Over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# df = pd.read_csv(
#     'final_results.csv'
# )

# # Use seaborn to compare the accuracy and number of parameters for each model
# sns.set(style="whitegrid")
# # sns.pairplot(df, hue="model", vars=["acc", "params"])
# sns.barplot(x="model", y="acc", data=df)


# plt.show()





# import tensorflow as tf

# class RandomFlipVideo(tf.keras.layers.Layer):
#   def __init__(self, **kwargs):
#     super(RandomFlipVideo, self).__init__()

#   @tf.function
#   def call(self, inputs):
#     return inputs
  
# class RandomRotationVideo(tf.keras.layers.Layer):
#   def __init__(self, max_rotation=0.3, **kwargs):
#     super(RandomRotationVideo, self).__init__()

#   @tf.function
#   def call(self, inputs):
#     return inputs

# model = tf.keras.models.load_model('Models/Violence_Acc_0.7350000143051147.h5', custom_objects={'RandomFlipVideo': RandomFlipVideo, 'RandomRotationVideo': RandomRotationVideo})

# model.summary()

# import cv2
# import numpy as np
# import pandas as pd
# import os
# import tensorflow as tf

# LABEL_DIR = 'Data/violence_bounding_box_labels.csv'
# DATA_DIR = 'Data/BoundingImages'

# labels = pd.read_csv(LABEL_DIR)
# labels['Path'] = labels['ImageID'].apply(lambda x: os.path.join(DATA_DIR, x))

# IMG_PATH = os.path.join(DATA_DIR, '1zcOeIvnGL97tfvK.avi_frame3.jpg')

# # Processing Functions
# @tf.function
# def preprocess(image, labels):
#     image = tf.image.decode_jpeg(image, channels=3)
#     # image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
#     # image = image / 255.0

#     return image, labels# * [image.shape[0] / image.shape[1], 1, image.shape[0] / image.shape[1], 1]

# @tf.function
# def load_and_preprocess_image(img_path, labels):
#     image = tf.io.read_file(img_path)
#     return preprocess(image, labels)

# # Data
# # paths = labels['Path']
# # df = labels[['YMin','XMin', 'YMax', 'XMax']]

# img_label = labels.loc[labels['Path'] == IMG_PATH]

# y1, x1, y2, x2 = img_label[['YMin','XMin', 'YMax', 'XMax']].values[0]

# image = cv2.imread(IMG_PATH)
# h, w = image.shape[:2]
# image = cv2.resize(image, (w * 2, h * 2))

# h, w = image.shape[:2]

# y1 = int(y1 * h)
# x1 = int(x1 * w)
# y2 = int(y2 * h)
# x2 = int(x2 * w)

# cv2.putText(image, 'Violence Detected: ', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# cv2.putText(image, '92%', (315, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# cv2.putText(image, 'Weapons: ', (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
# cv2.putText(image, 'None', (170, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# cv2.putText(image, 'Number Of People Involved:', (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
# cv2.putText(image, '2', (465, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# cv2.putText(image, 'Location: ', (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
# cv2.putText(image, '1234 Park Street', (160, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



# cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)



# cv2.imshow('image', image)
# cv2.waitKey(0)


# # Creating Dataset
# path_ds = tf.data.Dataset.from_tensor_slices(paths.to_numpy())

# label_ds = tf.data.Dataset.from_tensor_slices(df.to_numpy())

# image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))

# image_label_ds = image_label_ds.map(load_and_preprocess_image)
# image_label_ds = image_label_ds.shuffle(buffer_size=1000, reshuffle_each_iteration=False)


# for image, label in image_label_ds.take(5):
#     y1, x1, y2, x2 = label.numpy()
#     image = image.numpy()
#     image = image.astype(np.uint8)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     h, w = image.shape[:2]

#     y1 = int(y1 * h)
#     x1 = int(x1 * w)
#     y2 = int(y2 * h)
#     x2 = int(x2 * w)

#     cv2.putText(image, 'Violence Detected...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    

#     cv2.imshow('image', image)
#     cv2.waitKey(0)