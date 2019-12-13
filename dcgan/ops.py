import math
import numpy as np 
import tensorflow.compat.v1 as tf_v1

# from tensorflow.compat.v1.python.framework import ops

from dcgan.utils import *

try:
  image_summary = tf_v1.image_summary
  scalar_summary = tf_v1.scalar_summary
  histogram_summary = tf_v1.histogram_summary
except:
  image_summary = tf_v1.summary.image
  scalar_summary = tf_v1.summary.scalar
  histogram_summary = tf_v1.summary.histogram

if "concat_v2" in dir(tf_v1):
  def concat(tensors, axis, *args, **kwargs):
    return tf_v1.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf_v1.concat(tensors, axis, *args, **kwargs)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf_v1.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf_v1.layers.batch_normalization(x,
                      momentum=self.momentum,
                      epsilon=self.epsilon,
                      scale=True,
                      training=train)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf_v1.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf_v1.variable_scope(name):
    w = tf_v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf_v1.truncated_normal_initializer(stddev=stddev))
    conv = tf_v1.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf_v1.get_variable('biases', [output_dim], initializer=tf_v1.constant_initializer(0.0))
    conv = tf_v1.reshape(tf_v1.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf_v1.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf_v1.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf_v1.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf_v1.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf_v1.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf_v1.get_variable('biases', [output_shape[-1]], initializer=tf_v1.constant_initializer(0.0))
    deconv = tf_v1.reshape(tf_v1.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv
     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf_v1.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf_v1.variable_scope(scope or "Linear"):
    matrix = tf_v1.get_variable("Matrix", [shape[1], output_size], tf_v1.float32,
                 tf_v1.random_normal_initializer(stddev=stddev))
    bias = tf_v1.get_variable("bias", [output_size],
      initializer=tf_v1.constant_initializer(bias_start))
    if with_w:
      return tf_v1.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf_v1.matmul(input_, matrix) + bias
