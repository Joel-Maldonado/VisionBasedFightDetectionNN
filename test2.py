import tensorflow as tf

model = tf.keras.models.load_model('Models/Bounding_Box_GIoU_0.01907881535589695.h5', compile=False)

model.summary()

# Plot model
# tf.keras.utils.plot_model(model, to_file='BoundingBoxModel_PLOT.png')