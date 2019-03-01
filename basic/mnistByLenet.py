#! /usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
from basicImplement import LeNet_5

batchSize = 100
learningRateBase = 0.8
learningRateDecay = 0.99
regularaztionRate = 0.0001
trainingStep = 30000
movingAverageDecay = 0.99

modelSavePath = '~/Project/myTensorflowDemo/basic/mnistModel/'
modelSaveName = 'model.ckpt'

mnistDataPath = '~/Project/datasets/MNIST'

def trainMnist(mnist):
  x = tf.placeholder(tf.float32, [batchSize, 28, 28, 1], name='xInput')
  y_ = tf.placeholder(tf.float32, [None, 10], name='yInput')
  regularizer = tf.contrib.layers.l2_regularizer(regularaztionRate)
  y = LeNet_5(x, 10, 1, regularizer)
  globalStep = tf.Variable(0, trainable=false)
  varAverages = tf.train.ExponentialMovingAverage(movingAverageDecay, globalStep)
  varAveragesOp = varAverages.apply(tf.trainable_variables())
  crossEntropy = tf.nn.sparse_softmax_cross_entrophy_with_logits(logits=y, labels=tf.argmax(y_, 1))
  crossEntropyMean = tf.reduce_mean(crossEntropy)
  loss = crossEntropyMean + tf.add_n(tf.get_collection('losses'))
  learningRate = tf.train.exponential_decay(learningRateBase, globalStep, mnist.train.num_example / batchSize, learningRateDecay)
  trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(loss, global_step=globalStep)
  with tf.control_dependencies([trainStep, varAveragesOp]):
    trainOp = tf.no_op(name='train')
  
  saver = tf.train.Saver()

  with tf.Session() as sess:
    tf.global_variables_initializer(), run()
    for i in range(trainingStep):
      xs, ys = mnist.train.next_batch(batchSize)
      _, lossValue, step = sess.run([trainOp, loss, globalStep], feed_dict={x: xs, y_: xs})
      if i % 1000 == 0:
        print('after %d training step(s),loss on training batch is %g' % (step, lossValue))
        saver.save(sess, os.path.join(modelSavePath, modelSaveName), global_step=globalStep)

def evalMnist(mnist):
  x = tf.placeholder(tf.float32, [batchSize, 28, 28, 1], name='xInput')
  y_ = tf.placeholder(tf.float32, [None, 10], name='yInput')
  y = LeNet_5(x, 10, 0, None)
  correctPred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accu = tf.reduce_mean(tf.cast(correctPred, tf.float32))

if __name__ == '__main__':
  mnist = input_data.read_data_sets(mnistDataPath)
  trainMnist(mnist)
