import os
import tensorflow as tf

import sys

sys.path.append("../")

from networks import network_structure as mynet

import numpy as np

batch_size = 1

def main(argv):
    input_images = tf.placeholder(dtype=tf.float32,
            shape=(None, None, None, 3),
            name='input_images')

    model = mynet.StyleTransferModel(batch_size)
    model.build_forward_network(input_images)

    print("============Net Created============")
    # Save the graph
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        tf.train.write_graph(sess.graph_def, "./model_pb", "the_scream.pbtxt", as_text=True)

    print("Save done.")

if __name__ == "__main__":
    tf.app.run()
