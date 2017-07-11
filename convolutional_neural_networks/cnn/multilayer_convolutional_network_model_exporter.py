# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
#import numpy as np
import input_data

from tensorflow.contrib.session_bundle import exporter

import tensorflow as tf

#import cv2

FLAGS = None


#权重初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#卷积和池化
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main(_):
    # Import data
    mnist = input_data.read_data_sets('../dataset', one_hot=True)

    #img = cv2.imread("/Users/*****/Documents/TensorFlowTutorials/dataset/9.2694.jpg")
    #rows, cols = img.shape
    #imagePixes = rows * cols
    #img_arrayy = np.array(img).resize(imagePixes, 1)

    #feeddict = {placeholder: im_array}


    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    #第一层卷积
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    #第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    #密集连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    #Dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    #softmax层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



    #评估模型
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    #分类准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 生成saver
    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    # 会将已经保存的变量值resotre到变量中。
    saver.restore(sess, "../model/multilayer_convolutional_network")

    # Test trained model
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
            sess.graph.as_graph_def(),
            named_graph_signatures={
                'inputs': exporter.generic_signature({'x': x, 'keep_prob': keep_prob}),
                'outputs': exporter.generic_signature({'y': y_conv})})
    model_exporter.export(FLAGS.work_dir,
                              tf.constant(FLAGS.export_version),
                              sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../dataset',
                        help='Directory for storing input data')
    parser.add_argument('--work_dir', type=str, default='.',
                        help='dir')
    parser.add_argument('--export_version', type=str, default='1',
                        help='model')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
