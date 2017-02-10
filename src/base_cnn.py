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
            kernel_width = int(min([x._shape[2], kernel_size]))
            # Convolution Layer
            # W = tf.get_variable("W", shape=[num_filters*4, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            W = tf.Variable(tf.truncated_normal([kernel_size, kernel_width, int(x._shape[3]), num_filters], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
            c = tf.nn.conv2d(x, W, strides=stride, padding=padding, name="conv")
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

            return h

    def initInput(self, data_length, data_width, data_height, num_classes, num_filters, l2_reg_lambda=0.0, x=None, y=None):
        # Placeholders for input, output and dropout
        if x == None:
            self.input_x = tf.placeholder(tf.float32, [None, data_length, data_width, data_height], name="input_x")
        else:
            self.input_x = x

        if y == None:
            self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        else:
            self.input_y = y
        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
