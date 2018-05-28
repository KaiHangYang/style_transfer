import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.append("../")

from utils import tfrecord_reader
from networks import network_structure as mynet

################################# Parameters for training ################################
learning_rate = 0.003
batch_size = 4
input_img_size = 256
input_img_channels = 3
train_iteration = 160000 # two epoches

restore_prev_trained_models = False
style_image_name = "the_scream"

log_dir = "../logs/"
vgg16_checkpoint_path = "../models/vgg16/vgg_16.ckpt"
generator_checkpoint_path = "../models/generator/trained-0"
path_to_save_models = "../models/generator/trained_model-%s" % style_image_name
style_image_path = "../style_images/%s.jpg" % style_image_name

dataset_dir = "/home/kaihang/DataSet/style_transfer/"

# Mean pixel for all images in the set. It is provided by https://github.com/lengstrom/fast-style-transfer
dataset_means = np.array([123.68, 116.779, 103.939])

##########################################################################################
if not os.path.isdir(dataset_dir):
    print("DataSet directory is not existing!")
    quit()
tfrecord_list = [os.path.join(dataset_dir, i) for i in os.listdir(dataset_dir)]

if __name__ == "__main__":
    # use the default graph
    with tf.Graph().as_default():

        batch_imgs = tfrecord_reader.read_batch(tfrecord_list, batch_size = batch_size, is_shuffle=True, num_epochs=None)
        input_images = tf.placeholder(dtype=tf.float32,
                shape=[None, input_img_size, input_img_size, input_img_channels],
                name="input_images")
        input_styles = tf.placeholder(dtype=tf.float32,
                shape=[None, input_img_size, input_img_size, input_img_channels],
                name="input_styles")

        model = mynet.StyleTransferModel(batch_size)
        model.build_loss(input_images, input_styles, learning_rate)

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)

            training_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph, filename_suffix="-train")

            vgg16_restorer = tf.train.Saver(slim.get_variables_to_restore(include=["vgg_16"]))
            generator_saver = tf.train.Saver(slim.get_variables_to_restore(include=["generator"]), max_to_keep=20)

            # initialize the parameters first
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # restore vgg16 
            if not os.path.isfile(vgg16_checkpoint_path):
                print("VGG16 checkpoint file is not existing!")
                quit()
            else:
                vgg16_restorer.restore(sess, vgg16_checkpoint_path)

            # restore generator
            if restore_prev_trained_models:
                if not os.path.isfile(generator_checkpoint_path + ".data-0-"): # TODO waiting to change
                    print("Previously trained model is not existing!")
                    quit()
                else:
                    generator_saver.restore(sess, generator_checkpoint_path)

            # Read the style image
            if not os.path.isfile(style_image_path):
                print("Style image is not existing!")
                quit()
            else:
                style_image = cv2.imread(style_image_path)
                if (style_image.shape[0] < input_img_size or style_image.shape[1] < input_img_size) or (style_image.shape[0] > input_img_size and style_image.shape[1] > input_img_size):
                    min_size = min(style_image.shape[0], style_image.shape[1])
                    img_scale = 1.0 * input_img_size / min_size
                    style_image = cv2.resize(style_image, (int(img_scale * style_image.shape[1]), int(img_scale * style_image.shape[0])))

                pad_width = int((style_image.shape[1] - input_img_size) / 2.0)
                pad_height = int((style_image.shape[0] - input_img_size) / 2.0)

                style_image = style_image[pad_height:pad_height+input_img_size, pad_width:pad_width+input_img_size]

                batch_styles_np = np.zeros([batch_size, style_image.shape[0], style_image.shape[1], style_image.shape[2]], dtype=np.float32)
                for img_num in range(batch_size):
                    batch_styles_np[img_num] = style_image.copy().astype(np.float32)

            global_steps = 0

            while global_steps < train_iteration:
                batch_images_np = sess.run(batch_imgs)
                _, batch_loss_np, batch_content_loss_np, batch_style_loss_np, batch_summary = sess.run([
                model.train_op,
                model.loss,
                model.content_loss,
                model.style_loss,
                model.summary
                ],
                feed_dict = {
                    input_images: batch_images_np,
                    input_styles: batch_styles_np
                })

                print("######################### Iterations %d ##########################\n" % global_steps)
                print("Total loss: %0.8f\n" % batch_loss_np)
                print("Content loss: %0.8f\n" % batch_content_loss_np)
                print("Style loss: %0.8f\n" % batch_style_loss_np)
                training_writer.add_summary(batch_summary, global_steps)

                if global_steps % 5000 == 0:
                    generator_saver.save(sess, save_path=path_to_save_models, global_step = global_steps)

                global_steps += 1
            coord.request_stop()
            coord.join(threads)
