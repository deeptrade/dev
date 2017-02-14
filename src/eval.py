'''
Load the model we have trained, and then evaluate the a Yahoo Finance json file with the model.
'''
import argparse
import urllib
import os
import sys
import tensorflow as tf
import numpy as np
import json
import math
import datetime
import pdb
from prepare_data import getETFData
from stock_cnn import StockVGG
from stock_cnn import StockFCN
from stock_cnn import StockFC
from stock_cnn import StockSqueezeNet

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

def evaluate(x, y):
    with tf.Graph().as_default() as g:
        x_tensor = tf.constant(x)
        y_tensor = tf.constant(y)
        cnn = StockVGG(
            data_length=len(x[0]),
            data_width=len(x[0][0]),
            data_height=len(x[0][0][0]),
            num_classes=len(y[0]),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=0.0,
            x=x_tensor,
            y=y_tensor)

        # Calculate predictions.
        # top_k_op = tf.nn.in_top_k(cnn.scores, y_tensor, 1)

        # Restore the moving average version of the learned variables for eval.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            total_sample_count = len(y)
            feed_dict = {cnn.dropout_keep_prob: 1.0}
            probabilities, predictions = sess.run([cnn.softmax, cnn.predictions], feed_dict)

            i = 0
            correctCount = 0
            for truth in y:
                if truth[predictions[i]] == 1:
                    correctCount += 1
                else:
                    print("probability {} truth {}".format(probabilities[i], truth))
                i += 1
            print("\naccuracy: {}".format( float(correctCount) / float(len(y)) ))
            '''
            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
            '''
            print("done")

if __name__ == '__main__':
    tf.flags.DEFINE_string("input_filename", "", "The path to the json data file we want to evaluate, e.g. spy.json")
    tf.flags.DEFINE_integer("start_from", 2007, "Evaluate only data starting from this year (default 2007)")
    tf.flags.DEFINE_string("eval_dir", "", "The path to the directory we want to write the evaluate result, note - the directory will be overwritten")
    tf.flags.DEFINE_string("checkpoint_dir", "", "The path to the directory we want to load the checkpointed model data")
    tf.flags.DEFINE_integer("num_filters", 64, "Number of filters to begin with (default: 64), must match checkpoint config")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    currentDir = os.path.dirname(os.path.realpath(__file__))

    if len(FLAGS.input_filename) == 0:
        FLAGS.input_filename = currentDir+'/../data/spy.json'
    if len(FLAGS.checkpoint_dir) == 0:
        FLAGS.checkpoint_dir = currentDir+'/../output/checkpoints'
    if len(FLAGS.eval_dir) == 0:
        FLAGS.eval_dir = currentDir+'/../output/eval'
    x, y = getETFData(FLAGS.input_filename, FLAGS.start_from)

    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    
    evaluate(x, y)

