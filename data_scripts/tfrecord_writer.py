import numpy as np
import os
import sys
import tensorflow as tf
import cv2


# Default: one tfrecord contains 1w images, and I totally has 8 blocks
# Max image size 640 x 640
raw_max_size = 640

source_img_size = 256

class COCOWriter():
    def __init__(self, images_dir_path):
        img_list = os.listdir(images_dir_path)[0:80000]
        np.random.shuffle(img_list)

        self.img_list = [os.path.join(images_dir_path, i) for i in img_list]

    def writeTfrecord(self, target_dir):
        for block_num in range(8):
            cur_writer = tf.python_io.TFRecordWriter(os.path.join(target_dir, "block-%d.tfrecord" % block_num))

            cur_frame_num = 0
            for img_file in self.img_list[block_num * 10000: (block_num + 1) * 10000]:
                print("\rCurrently processing block(%d) frames(%08d)" % (block_num, cur_frame_num))
                cur_img = cv2.imread(img_file)

                raw_shape = cur_img.shape

                # cur_img = cv2.copyMakeBorder(cur_img, top=0, left=0, right = raw_max_size - cur_img.shape[1], bottom=raw_max_size - cur_img.shape[0], borderType=cv2.BORDER_CONSTANT, value=[128, 128, 128])
                cur_img = cv2.resize(cur_img, (source_img_size, source_img_size), interpolation=cv2.INTER_NEAREST)

                example = tf.train.Example(features = tf.train.Features(
                    feature={
                        "image": tf.train.Feature(bytes_list = tf.train.BytesList(value=[cur_img.tobytes()])),
                        # "width": tf.train.Feature(int64_list = tf.train.Int64List(value=[raw_shape[1]])),
                        # "height": tf.train.Feature(int64_list = tf.train.Int64List(value=[raw_shape[0]])),
                        }
                    ))

                cur_writer.write(example.SerializeToString())
                cur_frame_num += 1


if __name__ == "__main__":
    writer = COCOWriter("/home/kaihang/DataSet_2/train2014/")
    writer.writeTfrecord("/home/kaihang/DataSet/style_transfer/")
