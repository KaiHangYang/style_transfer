import numpy as np
import tensorflow as tf
import sys

# use the vgg16 contained by tensorflow slim
import tensorflow.contrib.slim.nets as nets

sys.path.append("../")


class StyleTransferModel():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    # Helper function to build convolution layer
    def _helper_conv2d(self, inputs, filter_num, kernel_size, strides, name, with_relu = False):
        # Default use relu for activation
        return tf.layers.conv2d(inputs=inputs,
                                filters=filter_num,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding="SAME",
                                activation=tf.nn.relu if with_relu else None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name=name)

    # Helper function to build instance normalization
    def _helper_instance_norm(self, inputs, name):
        with tf.variable_scope("InstanceNorm_" + name):
            e = 1e-9
            mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
            result = (inputs - mean) / tf.sqrt(var + e)

            return result

    # Helper function to build residual block
    def _helper_residual_block(self, inputs, filter_num, kernel_size, strides, name):
        with tf.variable_scope("ResidualBlock_" + name):
            conv1 = self._helper_conv2d(inputs, filter_num, kernel_size, strides, "conv1", True)
            conv2 = self._helper_conv2d(conv1, filter_num, kernel_size, strides, "conv2", False)
            residual_block = inputs + conv2
            return residual_block

    def _helper_resize_conv2d(self, inputs, filter_num, kernel_size, strides, name, is_training):
        with tf.variable_scope("ResizeConv2d_" + name)
            input_width = inputs.get_shape()[2].value if is_training else tf.shape(inputs)[2]
            input_height = inputs.get_shape()[1].value if is_training else tf.shape(inputs)[1]

            new_height = input_height * 2 * strides
            new_width = input_width * 2 * strides

            features_resized = tf.image.resize_images(inputs, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            return self._helper_conv2d(features_resized, filter_num, kernel_size, strides, "conv1")

    # for style loss compute
    def _helper_gram(self, inputs):
        

    def build_generator(self, images, training):
        images = tf.pad(images, [[0, 0], [10, 10], [10, 10], [0, 0]], model="REFLECT")
        with tf.variable_scope("generator"):
            conv1 = tf.nn.relu(self.__helper_instance_norm(self.__helper_conv2d(images, 32, 9, 1, "conv1")))
            conv2 = tf.nn.relu(self.__helper_instance_norm(self.__helper_conv2d(conv1, 64, 3, 2, "conv2")))
            conv3 = tf.nn.relu(self.__helper_instance_norm(self.__helper_conv2d(conv2, 128, 3, 2, "conv2")))
            residual1 = self._helper_residual_block(conv3, 128, 3, 1, "res1")
            residual2 = self._helper_residual_block(residual1, 128, 3, 1, "res2")
            residual3 = self._helper_residual_block(residual2, 128, 3, 1, "res3")
            residual4 = self._helper_residual_block(residual3, 128, 3, 1, "res4")
            residual5 = self._helper_residual_block(residual4, 128, 3, 1, "res5")

            resize_conv2d1 = tf.nn.relu(self._helper_instance_norm(self._helper_resize_conv2d(residual5, 64, 3, 1, "resize_conv1", is_training)))
            resize_conv2d2 = tf.nn.relu(self._helper_instance_norm(self._helper_resize_conv2d(resize_conv2d1, 32, 3, 1, "resize_conv2", is_training)))
            result_conv2d = tf.nn.tanh(self._helper_instance_norm(self._helper_conv2d(resize_conv2d2, 3, 9, 1, "result_conv")))

            result = (result_conv2d + 1.0) * 127.5

            height = tf.shape(result)[1]
            width = tf.shape(result)[2]
            result = tf.slice(result, [0, 10, 10, 0], [-1, height - 10, width - 10, -1])

            return result

    def build_lossnet(self, inputs):
        logits, endpoints_dict = nets.vgg.vgg_16(inputs, spatial_squeeze=False)
        return logits, endpoints_dict


