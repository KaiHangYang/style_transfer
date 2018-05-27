import numpy as np
import tensorflow as tf
import sys

# use the vgg16 contained by tensorflow slim
import tensorflow.contrib.slim.nets as nets
import tensorflow.contrib.slim as slim

sys.path.append("../")

class StyleTransferModel():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.content_layers = ["vgg_16/conv3/conv3_3"]
        self.style_layers = ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2", "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"]
        self.style_weight = 220.0
        self.content_weight = 1.0


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
        with tf.variable_scope("ResizeConv2d_" + name):
            input_width = inputs.get_shape()[2].value if is_training else tf.shape(inputs)[2]
            input_height = inputs.get_shape()[1].value if is_training else tf.shape(inputs)[1]

            new_height = input_height * 2 * strides
            new_width = input_width * 2 * strides

            features_resized = tf.image.resize_images(inputs, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            return self._helper_conv2d(features_resized, filter_num, kernel_size, strides, "conv1")

    # for style loss compute
    def _helper_gram(self, inputs):
        batch_shape = tf.shape(inputs)
        batch_size = batch_shape[0]
        height = batch_shape[1]
        width = batch_shape[2]
        feature_num = batch_shape[3]

        features = tf.reshape(inputs, [batch_size, -1, feature_num])
        grams = tf.matmul(features, features, transpose_a=True) / tf.to_float(width * height * feature_num)

        return grams

    def build_generator(self, images, training):
        images = tf.pad(images, [[0, 0], [10, 10], [10, 10], [0, 0]], mode="REFLECT")
        with tf.variable_scope("generator"):
            conv1 = tf.nn.relu(self._helper_instance_norm(self._helper_conv2d(images, 32, 9, 1, "conv1"), "conv1"))
            conv2 = tf.nn.relu(self._helper_instance_norm(self._helper_conv2d(conv1, 64, 3, 2, "conv2"), "conv2"))
            conv3 = tf.nn.relu(self._helper_instance_norm(self._helper_conv2d(conv2, 128, 3, 2, "conv3"), "conv3"))
            residual1 = self._helper_residual_block(conv3, 128, 3, 1, "res1")
            residual2 = self._helper_residual_block(residual1, 128, 3, 1, "res2")
            residual3 = self._helper_residual_block(residual2, 128, 3, 1, "res3")
            residual4 = self._helper_residual_block(residual3, 128, 3, 1, "res4")
            residual5 = self._helper_residual_block(residual4, 128, 3, 1, "res5")

            resize_conv2d1 = tf.nn.relu(self._helper_instance_norm(self._helper_resize_conv2d(residual5, 64, 3, 1, "resize_conv1", training), "resize_conv1"))
            resize_conv2d2 = tf.nn.relu(self._helper_instance_norm(self._helper_resize_conv2d(resize_conv2d1, 32, 3, 1, "resize_conv2", training), "resize_conv2"))
            result_conv2d = tf.nn.tanh(self._helper_instance_norm(self._helper_conv2d(resize_conv2d2, 3, 9, 1, "result_conv"), "resize_conv3"))

            result = (result_conv2d + 1.0) * 127.5

            height = tf.shape(result)[1]
            width = tf.shape(result)[2]
            result = tf.slice(result, [0, 10, 10, 0], [-1, height - 20, width - 20, -1])

            return result

    def build_lossnet(self, inputs):
        logits, endpoints_dict = nets.vgg.vgg_16(inputs, spatial_squeeze=False)
        return logits, endpoints_dict

    def build_content_loss(self, endpoints_mixed, content_layer_names):
        loss = 0
        for layer in content_layer_names:
            A, B, _ = tf.split(endpoints_mixed[layer], 3, 0)
            size = tf.size(A)
            loss += tf.nn.l2_loss(A - B) * 2 / tf.to_float(size)

        return loss

    def build_style_loss(self, endpoints_mixed, style_layer_names):
        loss = 0
        for layer in style_layer_names:
            _, B, C = tf.split(endpoints_mixed[layer], 3, 0)
            size = tf.size(B)
            loss += tf.nn.l2_loss(self._helper_gram(B) - self._helper_gram(C)) * 2 / tf.to_float(size)

        return loss


    def build_loss(self, images, style_images, lr):
        self.style_image = style_images
        self.inputs = images

        self.generated_images = self.build_generator(self.inputs, True)
        self.squeezed_generated_image = tf.image.encode_jpeg(tf.cast(tf.squeeze(self.generated_images, [0]), tf.uint8))

        _, endpoints_mixed = self.build_lossnet(tf.concat([self.inputs, self.generated_images, self.style_image], axis=0))

        self.content_loss = self.content_weight * self.build_content_loss(endpoints_mixed, self.content_layers)
        self.style_loss = self.style_weight * self.build_style_loss(endpoints_mixed, self.style_layers)
        self.loss = self.content_loss + self.style_loss

        tf.summary.scalar("losses/content_loss", self.content_loss)
        tf.summary.scalar("losses/style_loss", self.style_loss)
        tf.summary.scalar("losses/total_loss", self.loss)
        tf.summary.image("images/generated", self.generated_images)
        tf.summary.image("images/original", self.inputs)
        self.summary = tf.summary.merge_all()

        variable_for_training = slim.get_variables_to_restore(include=["generator"])
        gradients = tf.gradients(self.loss, variable_for_training)

        grad_and_var = list(zip(gradients, variable_for_training))

        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.apply_gradients(grads_and_vars=grad_and_var)
