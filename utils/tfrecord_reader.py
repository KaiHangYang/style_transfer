import numpy as np
import tensorflow as tf
import os

class PathError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg) + ": file is not existing!"


def read_and_decode(tfr_queue, img_size=256):
    tfr_reader = tf.TFRecordReader()
    _, serialized_example = tfr_reader.read(tfr_queue)

    features = tf.parse_single_example(serialized_example, features={
        "image": tf.FixedLenFeature([], tf.string)
        })


    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [img_size, img_size, 3])

    return [img]


def read_batch(tfr_paths, batch_size=1, is_shuffle=True, num_epochs=None):
    for tfr_file in tfr_paths:
        if not os.path.exists(tfr_file):
            raise PathError(tfr_file)

    with tf.name_scope("Images_Inputs"):
        tfr_queue = tf.train.string_input_producer(tfr_paths, num_epochs=num_epochs, shuffle=is_shuffle)
        data_list = [read_and_decode(tfr_queue) for _ in range(1 * len(tfr_paths))]


        if is_shuffle:
            batch_img = tf.train.shuffle_batch_join(data_list,
                                                     batch_size=batch_size,
                                                     capacity=400,
                                                     min_after_dequeue=80,
                                                     enqueue_many=False,
                                                     name='img_data_reader')

        else:
            batch_img = tf.train.batch_join(data_list,
                                             batch_size=batch_size,
                                             capacity=300,
                                             enqueue_many=False,
                                             name='img_data_reader')

    return batch_img
