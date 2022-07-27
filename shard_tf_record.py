import tensorflow as tf


RECORD_PATH = 'violence_rgb_opt_val.tfrecord'
OUT_DIR = 'violence_rgb_opt_val'

raw_dataset = tf.data.TFRecordDataset(RECORD_PATH)

shards = 10

for i in range(shards):
    writer = tf.data.experimental.TFRecordWriter(f"{RECORD_PATH.split('.')[0]}_{i}.tfrecord")
    writer.write(raw_dataset.shard(shards, i))
    print(i)