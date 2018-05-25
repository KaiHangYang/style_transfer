import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import tensorflow as tf
import numpy as np
import sys

sys.path.append("../")

from utils import tfrecord_reader


if __name__ == "__main__":
    tfrecord_list = ["/home/kaihang/DataSet/style_transfer/block-0.tfrecord"]
    batch_img, batch_img_size = tfrecord_reader.read_batch(tfrecord_list, batch_size=1, is_shuffle=False, num_epochs=None)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while True:
            img, img_size = sess.run([batch_img, batch_img_size])
            img = img[0, 0]
            img_size = img_size[0]
            cv2.imshow("test", img[0:img_size[1], 0:img_size[0]])
            cv2.waitKey()
