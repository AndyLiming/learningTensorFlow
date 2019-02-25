import tensorflow as tf
import numpy as np

def convLayer(input, outChannel, kSize, stride, padding):
  weights = tf.get_variable('weights',[kSize, kSize, input.shape[3].value, outChannel], initializer = tf.truncated_normal_initializer(stddev=0.1))
  bias = tf.get_variable('bias', [outChannel], initializer = tf.constant_initializer(0.1))
  conv = tf.nn.conv2d(input, weights, [1, stride, stride, 1], padding=padding)
  y = tf.nn.bias_add(conv, bias)
  #activedConv = tf.nn.relu(y)
  #return activedConv
  return y
#  tf.nn.get_variable(name, shape, initializer,regularizer,trainable，collections)

def LeNet_5(inputImg, numLabel, isTrain, regularizer):
  with tf.variable_scope('conv-1'):
    conv1 = convLayer(inputImg, 32, 5, 1, 'SAME')
    relu1 = tf.nn.relu(conv1)
  with tf.variable_scope('maxPool-1'):
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME')
  with tf.variable_scope('conv-2'):
    conv2 = convLayer(pool1, 64, 5, 1, 'SAME')
    relu2 = tf.nn.relu(conv2)
  with tf.variable_scope('maxPool-2'):
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME')
  poolShape = pool2.get_shape().as_list()
  nodesNum = poolShape[1] * poolShape[2] * poolShape[3]#poolShape[0] equals batch size
  reshaped = tf.reshape(pool2,[poolShape[0],nodesNum])
  with tf.variable_scope('fc-1'):
    fc1_weight = tf.get_variable('weight', [nodesNum, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
      tf.add_to_collection('losses', regularizer(fc1_weight))  # 全连接层权重加入正则化
    fc1_bias = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_bias)
    if isTrain: fc1 = tf.nn.dropout(fc1, 0.5)
  with tf.variable_scope('fc-2'):
    fc2_weight = tf.get_variable('weight', [512,numLabel], initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
      tf.add_to_collection('losses', regularizer(fc2_weight))  # 全连接层权重加入正则化
    fc2_bias = tf.get_variable('bias', [numLabel], initializer=tf.constant_initializer(0.1))
    logit = tf.matmul(fc1, fc2_weight) + fc2_bias
  return logit
