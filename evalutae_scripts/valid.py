import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.append("../")

from utils import tfrecord_reader
from networks import network_structure as mynet

################################# Parameters for training ################################
input_img_channels = 3
batch_size = 1
restore_prev_trained_models = True

generator_checkpoint_path = "../models/generators/trained_model-20000"

# Mean pixel for all images in the set. It is provided by https://github.com/lengstrom/fast-style-transfer
dataset_means = np.array([123.68, 116.779, 103.939])

##########################################################################################

if __name__ == "__main__":
    # use the default graph
    with tf.Graph().as_default():
        input_images = tf.placeholder(dtype=tf.float32,
                shape=[None, None, None, input_img_channels],
                name="input_images")

        model = mynet.StyleTransferModel(batch_size)
        model.build_forward_network(input_images)

        with tf.Session() as sess:

            generator_saver = tf.train.Saver(slim.get_variables_to_restore(include=["generator"]), max_to_keep=20)

            # initialize the parameters first
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # restore generator
            if restore_prev_trained_models:
                if not os.path.isfile(generator_checkpoint_path + ".data-00000-of-00001"): # TODO waiting to change
                    print("Previously trained model is not existing!")
                    quit()
                else:
                    generator_saver.restore(sess, generator_checkpoint_path)

            camera = cv2.VideoCapture(0)
            while True:
                ret, cur_img = camera.read()
                cur_img = cur_img[np.newaxis, :]
                generated_images = sess.run(model.generated_images, feed_dict={input_images: cur_img})

                cv2.imshow("raw_images", cur_img[0])
                cv2.imshow("styled_images", generated_images[0].astype(np.uint8))
                key = cv2.waitKey(4)

                if key == ord('q'):
                    break

