import tensorflow as tf
import numpy as np

def convLayer(input, outChannel, kSize, stride, padding):
  weights = tf.Variable(tf.truncated_normal([kSize, kSize, input.shape[3].value, outChannel], mean=0, stddev=0.1))
  bias = tf.Variabel(tf.zeros([outChannel]))
  conv = tf.nn.conv2d(input, weights, [1, stride, stride, 1], padding=padding)
  y = tf.nn.bias_add(conv, bias)
  activedConv = tf.nn.relu(y)
  return activedConv
