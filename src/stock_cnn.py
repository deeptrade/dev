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
        self.initInput(data_length, data_width, data_height, num_classes, num_filters, l2_reg_lambda, x, y)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        out = self.conv("conv1_1", self.input_x, 3, [1, 1, 1, 1], num_filters)
        out = self.conv("conv1_2", out, 3, [1, 1, 1, 1], num_filters)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1")

        out = self.conv("conv2_1", out, 3, [1, 1, 1, 1], num_filters*2)
        out = self.conv("conv2_2", out, 3, [1, 1, 1, 1], num_filters*2)
        out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool2")
        #out = tf.nn.dropout(out, self.dropout_keep_prob)

        out = self.conv("conv3_1", out, 3, [1, 1, 1, 1], num_filters*4)
        out = self.conv("conv3_2", out, 3, [1, 1, 1, 1], num_filters*4)
        out = self.conv("conv3_3", out, 3, [1, 1, 1, 1], num_filters*4)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool3")
        
        out = self.conv("conv4_1", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv4_2", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv4_3", out, 1, [1, 1, 1, 1], num_filters*8)
        out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
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
        out = self.fc("fc6", flat, flat_size, num_classes*32)
        out = tf.nn.dropout(out, self.dropout_keep_prob)
        out = self.fc("fc7", out, num_classes*32, num_classes*16)
        #out = tf.nn.dropout(out, self.dropout_keep_prob)
        out = self.fc("fc8", out, num_classes*16, num_classes, relu=False)
        l2_loss += tf.nn.l2_loss(self.publicVariables["fc8"]["W"])
        l2_loss += tf.nn.l2_loss(self.publicVariables["fc8"]["b"])
        self.scores = out

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
        self.initInput(data_length, data_width, data_height, num_classes, num_filters, l2_reg_lambda, x, y)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        out = self.conv("conv1_1", self.input_x, 3, [1, 1, 1, 1], num_filters)
        out = self.conv("conv1_2", out, 3, [1, 1, 1, 1], num_filters)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1")

        out = self.conv("conv2_1", out, 3, [1, 1, 1, 1], num_filters*2)
        out = self.conv("conv2_2", out, 3, [1, 1, 1, 1], num_filters*2)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool2")
        out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        #out = tf.nn.dropout(out, self.dropout_keep_prob)

        out = self.conv("conv3_1", out, 3, [1, 1, 1, 1], num_filters*4)
        out = self.conv("conv3_2", out, 3, [1, 1, 1, 1], num_filters*4)
        out = self.conv("conv3_3", out, 3, [1, 1, 1, 1], num_filters*4)
        out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool3")
        
        out = self.conv("conv4_1", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv4_2", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv4_3", out, 1, [1, 1, 1, 1], num_filters*8)
        out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool4")
        #out = tf.nn.dropout(out, self.dropout_keep_prob)

        '''
        out = self.conv("conv5_1", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv5_2", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv5_3", out, 1, [1, 1, 1, 1], num_filters*8)
        out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool5")
        '''
        out = tf.nn.dropout(out, self.dropout_keep_prob)

        # Combine all the pooled features
        flat_size = int(out._shape[1] * out._shape[2] * out._shape[3])
        flat = tf.reshape(out, [-1, flat_size])
        out = self.fc("fc6", flat, flat_size, num_classes, relu=False)
        l2_loss += tf.nn.l2_loss(self.publicVariables["fc6"]["W"])
        l2_loss += tf.nn.l2_loss(self.publicVariables["fc6"]["b"])
        self.scores = out

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
        self.initInput(data_length, data_width, data_height, num_classes, num_filters, l2_reg_lambda, x, y)

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
        out = self.fc("fc11", flat, flat_size, num_classes, relu=False)
        self.scores = out

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

class StockSquareFCN(BaseCNN):
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
        self.initInput(data_length, data_width, data_height, num_classes, num_filters, l2_reg_lambda, x, y)

        # Use FC layer to convert the input into a larger squre image
        out = self.input_x
        flat_size = int(data_length * data_width * data_height)
        flat = tf.reshape(out, [-1, flat_size])
        out = self.fc("fc0", flat, flat_size, int(flat_size * flat_size / 16))
        out = tf.reshape(out, [-1, int(flat_size/4), int(flat_size/4), 1])

        out = self.conv("conv1_1", out, 3, [1, 1, 1, 1], num_filters)
        out = self.conv("conv1_2", out, 3, [1, 1, 1, 1], num_filters)
        out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool1")

        out = self.conv("conv2_1", out, 3, [1, 1, 1, 1], num_filters*2)
        out = self.conv("conv2_2", out, 3, [1, 1, 1, 1], num_filters*2)
        out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool2")
        out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        #out = tf.nn.dropout(out, self.dropout_keep_prob)

        out = self.conv("conv3_1", out, 3, [1, 1, 1, 1], num_filters*4)
        out = self.conv("conv3_2", out, 3, [1, 1, 1, 1], num_filters*4)
        out = self.conv("conv3_3", out, 3, [1, 1, 1, 1], num_filters*4)
        out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
        out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool3")
        
        out = self.conv("conv4_1", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv4_2", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv4_3", out, 1, [1, 1, 1, 1], num_filters*8)
        out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
        out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool4")
        #out = tf.nn.dropout(out, self.dropout_keep_prob)

        out = self.conv("conv5_1", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv5_2", out, 1, [1, 1, 1, 1], num_filters*8)
        out = self.conv("conv5_3", out, 1, [1, 1, 1, 1], num_filters*8)
        out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool5")

        out = tf.nn.dropout(out, self.dropout_keep_prob)

        # Combine all the pooled features
        flat_size = int(out._shape[1] * out._shape[2] * out._shape[3])
        flat = tf.reshape(out, [-1, flat_size])
        out = self.fc("fc6", flat, flat_size, num_classes, relu=False)
        self.scores = out

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

class StockSquareSqueezeNet(BaseCNN):
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
        self.initInput(data_length, data_width, data_height, num_classes, num_filters, l2_reg_lambda, x, y)

        # Use FC layer to convert the input into a larger squre image
        out = self.input_x
        flat_size = int(data_length * data_width * data_height)
        flat = tf.reshape(out, [-1, flat_size])
        out = self.fc("fc0", flat, flat_size, int(flat_size * flat_size / 4))
        out = tf.reshape(out, [-1, int(flat_size/2), int(flat_size/2), 1])

        #pdb.set_trace()
        out = self.conv("conv1", out, 7, [1, 2, 2, 1], 96)
        out = tf.nn.max_pool(out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool1")

        out = self.fire("fire2", out, 16, 64)
        out = self.fire("fire3", out, 16, 64)
        out = self.fire("fire4", out, 32, 128)
        out = tf.nn.max_pool(out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool4")

        out = self.fire("fire5", out, 32, 128)
        out = self.fire("fire6", out, 48, 192)
        out = self.fire("fire7", out, 48, 192)
        out = self.fire("fire8", out, 64, 256)
        out = tf.nn.max_pool(out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool8")

        out = self.fire("fire9", out, 64, 256)
        out = tf.nn.dropout(out, self.dropout_keep_prob)

        '''
        out = self.conv("conv10_nexar", out, 1, [1, 1, 1, 1], num_classes)
        out = tf.nn.avg_pool(out, ksize=[1, int(out._shape[1]), int(out._shape[2]), 1], strides=[1, 1, 1, 1], padding='VALID', name="pool10")
        self.scores = tf.reshape(out, [-1, num_classes])
        '''
        
        flat_size = int(out._shape[1] * out._shape[2] * out._shape[3])
        flat = tf.reshape(out, [-1, flat_size])
        out = self.fc("fc11", flat, flat_size, num_classes, relu=False)
        self.scores = out

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

