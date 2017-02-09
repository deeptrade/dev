import tensorflow as tf
import numpy as np
import pdb
from base_cnn import BaseCNN

class StockVGG(BaseCNN):
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

        out = self.conv("conv1_1", self.input_x, 3, [1, 1, 1, 1], num_filters)
        out = self.conv("conv1_2", out, 3, [1, 1, 1, 1], num_filters)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1")

        out = self.conv("conv2_1", out, 3, [1, 1, 1, 1], num_filters*2)
        out = self.conv("conv2_2", out, 3, [1, 1, 1, 1], num_filters*2)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool2")
        #out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        #out = tf.nn.dropout(out, self.dropout_keep_prob)

        out = self.conv("conv3_1", out, 3, [1, 1, 1, 1], num_filters*4)
        out = self.conv("conv3_2", out, 3, [1, 1, 1, 1], num_filters*4)
        out = self.conv("conv3_3", out, 3, [1, 1, 1, 1], num_filters*4)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool3")
        
        out = self.conv("conv4_1", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv4_2", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv4_3", out, 1, [1, 1, 1, 1], num_filters*8)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool4")

        '''
        out = self.conv("conv5_1", out, 3, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv5_2", out, 3, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv5_3", out, 3, [1, 1, 1, 1], num_filters*8)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool5")
        '''

        # Combine all the pooled features
        flat_size = int(out._shape[1] * out._shape[2] * out._shape[3])
        flat = tf.reshape(out, [-1, flat_size])
        out = self.fc("fc6", flat, flat_size, num_classes*16)
        out = tf.nn.dropout(out, self.dropout_keep_prob)
        out = self.fc("fc7", out, num_classes*16, num_classes*16)
        out = tf.nn.dropout(out, self.dropout_keep_prob)
        out = self.fc("fc8", out, num_classes*16, num_classes, relu=False)
        l2_loss += tf.nn.l2_loss(self.publicVariables["fc8"]["W"])
        l2_loss += tf.nn.l2_loss(self.publicVariables["fc8"]["b"])
        self.scores = self.publicVariables["fc8"]["scores"]

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
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

class StockFCN(BaseCNN):
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

        out = self.conv("conv1_1", self.input_x, 3, [1, 1, 1, 1], num_filters)
        out = self.conv("conv1_2", out, 3, [1, 1, 1, 1], num_filters)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1")

        out = self.conv("conv2_1", out, 3, [1, 1, 1, 1], num_filters*2)
        out = self.conv("conv2_2", out, 3, [1, 1, 1, 1], num_filters*2)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool2")
        #out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        #out = tf.nn.dropout(out, self.dropout_keep_prob)

        out = self.conv("conv3_1", out, 3, [1, 1, 1, 1], num_filters*4)
        out = self.conv("conv3_2", out, 3, [1, 1, 1, 1], num_filters*4)
        out = self.conv("conv3_3", out, 3, [1, 1, 1, 1], num_filters*4)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool3")
        
        out = self.conv("conv4_1", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv4_2", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv4_3", out, 1, [1, 1, 1, 1], num_filters*8)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool4")
        out = tf.nn.dropout(out, self.dropout_keep_prob)

        out = self.conv("conv5_1", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv5_2", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv5_3", out, 1, [1, 1, 1, 1], num_filters*8)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool5")
        out = tf.nn.dropout(out, self.dropout_keep_prob)

        # Combine all the pooled features
        flat_size = int(out._shape[1] * out._shape[2] * out._shape[3])
        flat = tf.reshape(out, [-1, flat_size])
        out = self.fc("fc6", flat, flat_size, num_classes)
        l2_loss += tf.nn.l2_loss(self.publicVariables["fc6"]["W"])
        l2_loss += tf.nn.l2_loss(self.publicVariables["fc6"]["b"])
        self.scores = self.publicVariables["fc6"]["scores"]

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
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


class StockSqueezeNet(BaseCNN):
    def fire(self, name, x, squeeze_filters, expand_filters):
        with tf.name_scope(name):
            squeeze = self.conv("squeeze1x1", x, 1, [1, 1, 1, 1], squeeze_filters)
            expand1 = self.conv("expand1x1", squeeze, 1, [1, 1, 1, 1], expand_filters)
            expand3 = self.conv("expand3x3", squeeze, 3, [1, 1, 1, 1], expand_filters)
            out = tf.concat_v2([expand1, expand3], 3, name="concat")

            self.publicVariables[name] = {}
            self.publicVariables[name]["squeeze1x1"] = self.publicVariables["squeeze1x1"]["W"]
            self.publicVariables[name]["expand1x1"] = self.publicVariables["expand1x1"]["W"]
            self.publicVariables[name]["expand3x3"] = self.publicVariables["expand3x3"]["W"]
            return out

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

        #pdb.set_trace()
        out = self.conv("conv1", self.input_x, 7, [1, 2, 1, 1], 96)
        out = tf.nn.max_pool(out, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1")

        out = self.fire("fire2", out, 16, 64)
        out = self.fire("fire3", out, 16, 64)
        out = self.fire("fire4", out, 32, 128)
        out = tf.nn.max_pool(out, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool4")

        out = self.fire("fire5", out, 32, 128)
        out = self.fire("fire6", out, 48, 192)
        out = self.fire("fire7", out, 48, 192)
        out = self.fire("fire8", out, 64, 256)
        out = tf.nn.max_pool(out, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool8")

        out = self.fire("fire9", out, 64, 256)
        out = tf.nn.dropout(out, self.dropout_keep_prob)
        out = self.conv("conv10_nexar", out, 1, [1, 1, 1, 1], num_classes)
        
        # out = tf.nn.avg_pool(out, ksize=[1, int(out._shape[1]), 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool10")
        # self.scores = tf.reshape(out, [-1, num_classes])

        flat_size = int(out._shape[1] * out._shape[2] * out._shape[3])
        flat = tf.reshape(out, [-1, flat_size])
        out = self.fc("fc11", flat, flat_size, num_classes)
        self.scores = self.publicVariables["fc11"]["scores"]

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.softmax = tf.nn.softmax(self.scores)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
