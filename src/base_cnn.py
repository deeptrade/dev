import tensorflow as tf
import numpy as np

class BaseCNN(object):
    # dict that holds the variables that we are interested in exposing        
    publicVariables = {}
    '''
    create a conv + relu layer
    '''
    def conv(self, name, x, kernel_size, stride, num_filters, padding="SAME"):
        with tf.name_scope(name):
            # Convolution Layer
            # W = tf.get_variable("W", shape=[num_filters*4, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            W = tf.Variable(tf.truncated_normal([kernel_size, int(x._shape[2]), int(x._shape[3]), num_filters], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
            c = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding, name="conv")
            # Relu
            h = tf.nn.relu(tf.nn.bias_add(c, b), name="relu")

            self.publicVariables[name] = {}
            self.publicVariables[name]["W"] = W
            self.publicVariables[name]["b"] = b
            return h

    def fc(self, name, x, data_width, node_count, relu=True):
        with tf.name_scope(name):
            W = tf.Variable(tf.truncated_normal([data_width, node_count], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[node_count]), name="b")
            c = tf.nn.xw_plus_b(x, W, b, name="scores")
            if relu:
                h = tf.nn.relu(c, name="relu")
            else:
                h = c
            self.publicVariables[name] = {}
            self.publicVariables[name]["W"] = W
            self.publicVariables[name]["b"] = b
            self.publicVariables[name]["scores"] = c
            return h
