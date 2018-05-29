import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse

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
model_dir_path = "../models/stable_models/"
##########################################################################################

if __name__ == "__main__":
    # use the default graph
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-l", "--list", help="List the availiable styles", action="store_true")
    arg_parser.add_argument("-s", "--style", help="Use the style_name")
    arg_parser.add_argument("-m", "--mode", help="Defualt 0. If type is 0 then you need input the image directory to read to transfer, if type is 1 then use webcam as the input source if type is 2 then use the video as the input source", type=int, choices=[0, 1, 2], default=0)
    arg_parser.add_argument("-i", "--input_dir", help="Only useful when mode is 0. The input image directory.")
    arg_parser.add_argument("-o", "--output_dir", help="Only useful when mode is 0. The output image directory.")
    arg_parser.add_argument("-n", "--img_num", help="Default 10. Only useful when mode is 0. The max image num selected from the input directory.", type=int, default=10)
    arg_parser.add_argument("-iv", "--input_video", help="Only useful when mode is 2. The input video path.")
    arg_parser.add_argument("-r", "--resize", help="Resize the input image to 256x256 if setted.", action="store_true")


    args = arg_parser.parse_args()

    if not os.path.isdir(model_dir_path) or len(os.listdir(model_dir_path)) == 0:
        print("You need to put the style models in $PROJECT_DIR/models/stable_models/")
        quit()

    if args.list:
        print("The availiable style list:")
        style_list = os.listdir(model_dir_path)
        for i in range(len(style_list)):
            print("style name:\t" + style_list[i])
        quit()

    if args.style is None:
        print("Use -h or --help to see the usage.")
        quit()

    used_mode = args.mode
    is_resize = args.resize

    if used_mode == 0:
        if args.input_dir is None or args.output_dir is None:
            print("Use -h or --help to see the usage.")
            quit()

        input_dir = args.input_dir
        output_dir = args.output_dir
        test_img_num = args.img_num

        if test_img_num <= 0:
            print("The image num must be large than 0")
            quit()
        if not os.path.isdir(input_dir):
            print("The input directory is not existing!")
            quit()
        if not os.path.isdir(output_dir):
            print("The output directory is not existing!")
            quit()

        input_image_list = np.array(os.listdir(input_dir))

        if len(input_image_list) == 0:
            print("The input directory contains no files!")
            quit()

        if len(input_image_list) < test_img_num:
            test_img_num = len(input_image_list)
        selected_arr = np.random.shuffle(np.arange(0, len(input_image_list), 1))[0:test_img_num]
        input_image_list_raw = input_image_list[selected_arr]

        input_image_list = [os.path.join(input_dir, i) for i in input_image_list_raw]
        output_image_list = [os.path.join(output_dir, "styled_" + i) for i in input_image_list_raw]

    elif used_mode == 2:
        if args.input_video is None or not os.path.isfile(args.input_video):
            print("You need to use -iv to specify the input video path.")
            quit()

        camera = cv2.VideoCapture(args.input_video)

        if not camera.isOpened():
            print("The video is not valid or your opencv doesn't support VideoCapture!")
            quit()

    elif used_mode == 1:
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("Your webcam is not valid or your opencv doesn't support VideoCapture!")
            quit()

    generator_checkpoint_path = os.path.join(model_dir_path, "%s/%s" % (args.style, args.style))

    with tf.Graph().as_default():
        input_images = tf.placeholder(dtype=tf.float32,
                shape=[None, None, None, input_img_channels],
                name="input_images")

        model = mynet.StyleTransferModel(batch_size)
        model.build_forward_network(input_images)

        with tf.Session() as sess:

            generator_saver = tf.train.Saver(slim.get_variables_to_restore(include=["generator"]))

            # initialize the parameters first
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # restore generator
            if not os.path.isfile(generator_checkpoint_path + ".data-00000-of-00001"): # TODO waiting to change
                print("Style is not existing!")
                quit()
            else:
                generator_saver.restore(sess, generator_checkpoint_path)

            cur_index = 0
            while True:

                if used_mode == 0:
                    if cur_index >= len(input_image_list):
                        break
                    used_index = cur_index
                    cur_img = cv2.imread(input_image_list[used_index])
                    cur_index += 1

                elif used_mode == 1 or used_mode == 2:
                    ret, cur_img = camera.read()

                if cur_img is None:
                    if used_mode == 0:
                        print("Some file is not image in the input directory!")
                        quit()
                    else:
                        print("Finished!")
                        break

                if is_resize:
                    cur_img = cv2.resize(cur_img, (256 ,256))

                cur_img = cur_img[np.newaxis, :]
                generated_images = sess.run(model.generated_images, feed_dict={input_images: cur_img})

                cv2.imshow("raw_images", cur_img[0])
                cv2.imshow("styled_images", generated_images[0].astype(np.uint8))
                key = cv2.waitKey(4)

                if used_mode == 0:
                    cv2.imwrite(output_image_list[used_index], generated_images[0].astype(np.uint8))

                if key == ord('q'):
                    break

