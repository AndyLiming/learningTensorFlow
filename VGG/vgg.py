import os
import time

import tensorflow as tf
import numpy as np

class vgg16:
  def __init__(self, imgs, weights=None,train = false):
    self.imgs = imgs
    self.build(imgs, train)
    self.probs = tf.nn.softmax(self.fc3)

  def convLayer(self, input, name,outChannel):
    with tf.variable_scope(name) as scope:
      weights = tf.get_variable(name + '_weights', shape=[3, 3, input.shape[3].value, outChannel], initializer=tf.truncated_normal_initializer())#kernel
      biases = tf.get_variable(name + '_biases', shape=[outChannel], initializer=tf.constant_initializer())#bias
      conv = tf.nn.conv2d(input,weights,padding = 'SAME')#build conv kernels
      add_bias = tf.nn.bias_add(conv, biases)#add biases
      out = tf.nn.relu(add_bias)#activity function
      return out
  def maxPooling(self, input, name):
    with tf.variable_scope(name) as scope:
      out = tf.nn.max_pool(name + '_maxpooling', input, shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      return out
  def fcLayer(self, input, name, outNum):
    with tf.variable_scope(name) as scope:
      shape = input.get_shape().as_list()
      dim = 1
      for d in shape[1:]:
        dim *= d
      x = tf.reshape(input, [-1, dim])
      weights = tf.get_variable(name + '_weights', shape=[dim, outNum] initializer=tf.truncated_normal_initializer())
      bias = tf.get_variable(name + '_biases', shape=[outNum], initializer=tf.constant_initializer())
      fc = tf.nn.bias_add(tf.matmul(x, weights), bias)
      return fc
  
  def build(self, bgrImage, train=false):
    #require input image: BGR 0-255
    assert bgrImage.get_shape().as_list()[1:] == [224, 224, 3]
    #
    self.conv1_1 = self.convLayer(bgr, 'conv1_1', 64)
    self.conv1_2 = self.convLayer(self.conv1_1, 'conv1_2', 64)
    self.pool1 = self.maxPooling(self.conv1_2, 'pool1')
    #
    self.conv2_1 = self.convLayer(self.pool1, 'conv2_1', 128)
    self.conv2_2 = self.convLayer(self.conv2_1, 'conv2_2', 128)
    self.pool2 = self.maxPooling(self.conv2_2, 'pool2')
    #
    self.conv3_1 = self.convLayer(pool2, 'conv3_1', 256)
    self.conv3_2 = self.convLayer(self.conv3_1, 'conv3_2', 256)
    self.conv3_3 = self.convLayer(self.conv3_2, 'conv3_3', 256)
    self.pool3 = self.maxPooling(self.conv3_3, 'pool3')
    #
    self.conv4_1 = self.convLayer(pool3, 'conv4_1', 512)
    self.conv4_2 = self.convLayer(self.conv4_1, 'conv4_2', 512)
    self.conv4_3 = self.convLayer(self.conv4_2, 'conv4_3', 512)
    self.pool4 = self.maxPooling(self.conv4_3, 'pool4')
    #
    self.conv5_1 = self.convLayer(pool4, 'conv5_1', 512)
    self.conv5_2 = self.convLayer(self.conv5_1, 'conv5_2', 512)
    self.conv5_3 = self.convLayer(self.conv5_2, 'conv5_3', 512)
    self.pool5 = self.maxPooling(self.conv5_3, 'pool5')
    #
    self.fc1 = self.fcLayer(self.pool5, 'fc1', 4096)
    if train:fc1 = tf.nn.dropout(fc1,0.5)
    self.fc2 = self.fcLayer(self.fc1, 'fc1', 4096)
    if train:fc2 = tf.nn.dropout(fc2,0.5)
    self.fc3 = self.fcLayer(self.fc2, 'fc1', 1000)
    if train:fc3 = tf.nn.dropout(fc3,0.5)

if __name__ == '__main__':