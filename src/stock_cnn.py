import tensorflow as tf
import numpy as np

class StockCNN(object):
    '''
    create a conv + relu layer
    '''
    def createConv(data_length, data_width, data_height, kernel_size, stride, num_filters):
        return 0

    """
    A CNN for stock classification.
    Uses a convolutional, max-pooling and softmax layer.
    data_length is the number of dates we have in one sample of data.
    data_width is the number of data points we have in each date.
    num_classes is the number of prediction classes
    filter_sizes is an array of filter size we want to use along the data_length direction
    num_filters is the number of filters we use for each filter size
    """
    def __init__(self, data_length, data_width, data_height, num_classes, num_filters, l2_reg_lambda=0.0, x=None, y=None):

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

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Create a convolution + maxpool layer 
        pooled_outputs = []
        with tf.name_scope("conv-maxpool-1"):
            # Convolution Layer
            W = tf.Variable(tf.truncated_normal([3, data_width, data_height, num_filters], stddev=0.1), name="W1")
            b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b1")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
            # Relu
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu1")
            self.w1 = W

            # conv again
            W = tf.Variable(tf.truncated_normal([3, 1, num_filters, num_filters], stddev=0.1), name="W2")
            b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b2")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu2")

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool")
            #dropped = tf.nn.dropout(pooled, self.dropout_keep_prob)
            #norm1 = tf.nn.lrn(dropped, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')

        with tf.name_scope("conv-maxpool-2"):
            # Convolution Layer
            W = tf.Variable(tf.truncated_normal([3, 1, num_filters, num_filters*2], stddev=0.1), name="W1")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters*2]), name="b1")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
            # Relu
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu1")

            W = tf.Variable(tf.truncated_normal([3, 1, num_filters*2, num_filters*2], stddev=0.1), name="W2")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters*2]), name="b2")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
            # Relu
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu2")

            pooled = tf.nn.max_pool(h, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1],
                padding='VALID', name="pool")
            # Add dropout
            #dropped = tf.nn.dropout(pooled, self.dropout_keep_prob)
            #norm2 = tf.nn.lrn(dropped, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')

        with tf.name_scope("conv-maxpool-3"):
            # Convolution Layer, reduces the data_length to 1
            W = tf.Variable(tf.truncated_normal([int(data_length/4), 1, num_filters*2, num_filters*4], stddev=0.1), name="W1")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters*4]), name="b1")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
            # Relu
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu1")

            W = tf.Variable(tf.truncated_normal([1, 1, num_filters*4, num_filters*4], stddev=0.1), name="W2")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters*4]), name="b2")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
            # Relu
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu2")

            # Add dropout, no max pooling needed since it's at 1x1 now
            #dropped = tf.nn.dropout(h, self.dropout_keep_prob)
            #norm3 = tf.nn.lrn(dropped, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')

        # Combine all the pooled features
        self.h_pool_flat = tf.reshape(h, [-1, num_filters*4])

        # FC
        with tf.name_scope("FC"):
            W = tf.Variable(tf.truncated_normal([num_filters*4, 2048], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[2048]), name="b")
            h = tf.nn.xw_plus_b(self.h_pool_flat, W, b, name="scores")
            dropped = tf.nn.dropout(h, self.dropout_keep_prob)
            self.wfc = W

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # W = tf.get_variable("W", shape=[num_filters*4, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            W = tf.Variable(tf.truncated_normal([2048, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(dropped, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.softmax = tf.nn.softmax(self.scores)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
