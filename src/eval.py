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
from prepare_data import getETFData
from stock_cnn import StockCNN

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

def evaluate(x, y):
    with tf.Graph().as_default() as g:
        cnn = StockCNN(
            data_length=len(x[0]),
            data_width=len(x[0][0]),
            num_classes=len(y[0]),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=0.0)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(cnn.scores, y, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

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
            feed_dict = {
                cnn.input_x: x,
                cnn.input_y: y,
                cnn.dropout_keep_prob: 1.0
            }
            predictions = sess.run([top_k_op], feed_dict)
            true_count = np.sum(predictions)

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)

if __name__ == '__main__':
    tf.flags.DEFINE_string("input_filename", "", "The path to the json data file we want to evaluate, e.g. spy.json")
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
    x, y = getETFData(FLAGS.input_filename)

    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    
    evaluate(x,y)

