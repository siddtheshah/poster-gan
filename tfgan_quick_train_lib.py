import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
# Allow matplotlib images to render immediately.
tf.logging.set_verbosity(tf.logging.ERROR)  # Disable noisy outputs.

train_batch_size = 32 #@param
noise_dimensions = 64 #@param
generator_lr = 0.001 #@param
discriminator_lr = 0.0002 #@param


def input_fn(mode, params, shuffle_control=False):
    assert 'batch_size' in params
    assert 'noise_dims' in params
    bs = params['batch_size']
    nd = params['noise_dims']
    split = 'train' if mode == tf.estimator.ModeKeys.TRAIN else 'test'
    shuffle = shuffle_control and (mode == tf.estimator.ModeKeys.TRAIN)
    just_noise = (mode == tf.estimator.ModeKeys.PREDICT)

    noise_ds = (tf.data.Dataset.from_tensors(0).repeat()
                .map(lambda _: tf.random_normal([bs, nd])))

    if just_noise:
        return noise_ds

def _dense(inputs, units, l2_weight):
  return tf.layers.dense(
      inputs, units, None,
      kernel_initializer=tf.keras.initializers.glorot_uniform,
      kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
      bias_regularizer=tf.keras.regularizers.l2(l=l2_weight))

def _batch_norm(inputs, is_training):
  return tf.layers.batch_normalization(
      inputs, momentum=0.999, epsilon=0.001, training=is_training)

def _deconv2d(inputs, filters, kernel_size, stride, l2_weight):
  return tf.layers.conv2d_transpose(
      inputs, filters, [kernel_size, kernel_size], strides=[stride, stride],
      activation=tf.nn.relu, padding='same',
      kernel_initializer=tf.keras.initializers.glorot_uniform,
      kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
      bias_regularizer=tf.keras.regularizers.l2(l=l2_weight))

def _conv2d(inputs, filters, kernel_size, stride, l2_weight):
  return tf.layers.conv2d(
      inputs, filters, [kernel_size, kernel_size], strides=[stride, stride],
      activation=None, padding='same',
      kernel_initializer=tf.keras.initializers.glorot_uniform,
      kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
      bias_regularizer=tf.keras.regularizers.l2(l=l2_weight))


def unconditional_generator(noise, mode, weight_decay=2.5e-5):
    """Generator to produce unconditional MNIST images."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    net = _dense(noise, 1024, weight_decay)
    net = _batch_norm(net, is_training)
    net = tf.nn.relu(net)

    net = _dense(net, 7 * 7 * 256, weight_decay)
    net = _batch_norm(net, is_training)
    net = tf.nn.relu(net)

    net = tf.reshape(net, [-1, 7, 7, 256])
    net = _deconv2d(net, 64, 4, 2, weight_decay)
    net = _deconv2d(net, 64, 4, 2, weight_decay)
    # Make sure that generator output is in the same range as `inputs`
    # ie [-1, 1].
    net = _conv2d(net, 1, 4, 1, 0.0)
    net = tf.tanh(net)

    return net

def _leaky_relu(net):
    return tf.nn.leaky_relu(net, alpha=0.01)

def unconditional_discriminator(img, unused_conditioning, mode, weight_decay=2.5e-5):
    del unused_conditioning
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    net = _conv2d(img, 64, 4, 2, weight_decay)
    net = _leaky_relu(net)

    net = _conv2d(net, 128, 4, 2, weight_decay)
    net = _leaky_relu(net)

    net = tf.layers.flatten(net)

    net = _dense(net, 1024, weight_decay)
    net = _batch_norm(net, is_training)
    net = _leaky_relu(net)

    net = _dense(net, 1, weight_decay)

    return net

def get_eval_metric_ops_fn(gan_model):
  real_data_logits = tf.reduce_mean(gan_model.discriminator_real_outputs)
  gen_data_logits = tf.reduce_mean(gan_model.discriminator_gen_outputs)

  #TODO: Set up eval metrics
  return {
      'real_data_logits': tf.metrics.mean(real_data_logits),
      'gen_data_logits': tf.metrics.mean(gen_data_logits),
  }

def gen_opt():
  gstep = tf.train.get_or_create_global_step()
  base_lr = generator_lr
  # Halve the learning rate at 1000 steps.
  lr = tf.cond(gstep < 1000, lambda: base_lr, lambda: base_lr / 2.0)
  return tf.train.AdamOptimizer(lr, 0.5)

