import tensorflow as tf
import numpy as np

def convLayer(input, outChannel, kSize, stride, padding):
  weights = tf.Variable(tf.truncated_normal([kSize, kSize, input.shape[3].value, outChannel], mean=0, stddev=0.1))
  bias = tf.Variabel(tf.zeros([outChannel]))
  conv = tf.nn.conv2d(input, weights, [1, stride, stride, 1], padding=padding)
  y = tf.nn.bias_add(conv, bias)
  #activedConv = tf.nn.relu(y)
  #return activedConv
  return y

def LeNet_5(inputImg, numLabel, isTrain, regularizer):
  with tf.variable_scope('conv-1'):
    conv1 = convLayer(inputImg, 32, 5, 1, 'SAME')
    relu1 = tf.nn.relu(conv1)
  with tf.variable_scope('maxPool-1'):
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME')
  with tf.variable_scope('conv-2'):
    conv2 = convLayer(pool1, 64, 5, 1, 'SAME')
    relu1 = tf.nn.relu(conv2)
  with tf.variable_scope('maxPool-2'):
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME')
  
