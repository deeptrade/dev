import tensorflow as tf
import numpy as np

class StockCNN(object):
    """
    A CNN for stock classification.
    Uses a convolutional, max-pooling and softmax layer.
    data_length is the number of dates we have in one sample of data.
    data_width is the number of data points we have in each date.
    num_classes is the number of prediction classes
    filter_sizes is an array of filter size we want to use along the data_length direction
    num_filters is the number of filters we use for each filter size
    """
    def __init__(self, data_length, data_width, num_classes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, data_length, data_width, 1], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Create a convolution + maxpool layer 
        pooled_outputs = []
        with tf.name_scope("conv-maxpool-1"):
            # Convolution Layer
            W = tf.Variable(tf.truncated_normal([3, data_width, 1, num_filters], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")

            # Relu
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1],
                padding='SAME', name="pool")
            # Add dropout
            dropped = tf.nn.dropout(pooled, self.dropout_keep_prob)

            norm1 = tf.nn.lrn(dropped, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                name='norm')

        with tf.name_scope("conv-maxpool-2"):
            # Convolution Layer
            W = tf.Variable(tf.truncated_normal([3, 1, num_filters, num_filters], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(norm1, W, strides=[1, 1, 1, 1], padding="SAME", name="conv")

            # Relu
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            # Maxpooling over the outputs. Hardcoded for now. We know with this pooling we reduce it to 1 number per feature.
            pooled = tf.nn.max_pool(h, ksize=[1, 9, 1, 1], strides=[1, 1, 1, 1],
                padding='VALID', name="pool")
            # Add dropout
            dropped = tf.nn.dropout(pooled, self.dropout_keep_prob)

            norm2 = tf.nn.lrn(dropped, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                name='norm')

        # Combine all the pooled features
        self.h_pool_flat = tf.reshape(norm2, [-1, num_filters])

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_filters, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_pool_flat, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")