import tensorflow as tf

model = tf.keras.models.load_model(
  'Models/Bounding_Box_GIoU_0.026832809671759605.h5'
)

# model.summary()
tf.keras.utils.plot_model(model, to_file='bounding_box_giou_summary.png')