import tensorflow.compat.v1 as tf_v1

# Allow matplotlib images to render immediately.
tf_v1.logging.set_verbosity(tf_v1.logging.ERROR)  # Disable noisy outputs.

train_batch_size = 32 #@param
noise_dimensions = 64 #@param
generator_lr = 0.001 #@param
discriminator_lr = 0.0002 #@param

def _dense(inputs, units, l2_weight):
  return tf_v1.layers.dense(
      inputs, units, None,
      kernel_initializer=tf_v1.keras.initializers.glorot_uniform,
      kernel_regularizer=tf_v1.keras.regularizers.l2(l=l2_weight),
      bias_regularizer=tf_v1.keras.regularizers.l2(l=l2_weight))

def _batch_norm(inputs, is_training):
  return tf_v1.layers.batch_normalization(
      inputs, momentum=0.999, epsilon=0.001, training=is_training)

def _conv2d(x, filters, kernel_size, stride):
  return tf_v1.layers.conv2d(
      x,
      filters, [kernel_size, kernel_size],
      strides=[stride, stride],
      padding='same',
      kernel_initializer=tf_v1.initializers.truncated_normal(
          stddev=0.02))


def _deconv2d(x, filters, kernel_size, stride):
  return tf_v1.layers.conv2d_transpose(
      x,
      filters, [kernel_size, kernel_size],
      strides=[stride, stride],
      padding='same',
      kernel_initializer=tf_v1.initializers.truncated_normal(
          stddev=0.02))

def unconditional_generator(noise, mode=True, weight_decay=2.5e-5):
    """Generator to produce unconditional MNIST images."""
    is_training = (mode == tf_v1.estimator.ModeKeys.TRAIN)

    net = _dense(noise, 1024, weight_decay)
    net = _batch_norm(net, is_training)
    net = tf_v1.nn.relu(net)

    # 8 neurons per channel, 8 layers
    net = _dense(net, 8 * 8 * 256, weight_decay)
    net = _batch_norm(net, is_training)
    net = tf_v1.nn.relu(net)

    net = tf_v1.reshape(net, [-1, 8, 8, 256])

    # Expanding into 128 and then 256 channels
    net = _deconv2d(net, 128, 4, 2)
    net = _deconv2d(net, 256, 4, 2)
    # Make sure that generator output is in the same range as `inputs`
    # ie [-1, 1].

    # Flatten into 3 channel image
    net = _deconv2d(net, 3, 4.0, 2)
    net = tf_v1.tanh(net)

    net = 127.5*(net+1)
    #net = tf_v1.dtypes.cast(net, dtype=tf_v1.uint8)

    print("Generator shape", net.shape)
    print("Generator created")
    return net

def _leaky_relu(net):
    return tf_v1.nn.leaky_relu(net, alpha=0.01)

def unconditional_discriminator(img, unused_conditioning, mode, weight_decay=2.5e-5):
    is_training = (mode == tf_v1.estimator.ModeKeys.TRAIN)
    print("Discriminator image shape: ", img.shape)

    # Decompose into 256 channels, then 128 channels.
    net = _conv2d(img, 256, 4, 2)
    net = _leaky_relu(net)

    net = _conv2d(net, 128, 4, 2)
    net = _leaky_relu(net)

    net = tf_v1.layers.flatten(net)

    net = _dense(net, 1024, weight_decay)
    net = _batch_norm(net, is_training)
    net = _leaky_relu(net)

    net = _dense(net, 1, weight_decay)
    print("Discriminator created")
    return net

def get_eval_metric_ops_fn(gan_model):
  real_data_logits = tf_v1.reduce_mean(gan_model.discriminator_real_outputs)
  gen_data_logits = tf_v1.reduce_mean(gan_model.discriminator_gen_outputs)

  #TODO: Set up eval metrics
  return {
      'real_data_logits': tf_v1.metrics.mean(real_data_logits),
      'gen_data_logits': tf_v1.metrics.mean(gen_data_logits),
  }

def gen_opt():
  gstep = tf_v1.train.get_or_create_global_step()
  base_lr = generator_lr
  # Halve the learning rate at 1000 steps.
  lr = tf_v1.cond(gstep < 1000, lambda: base_lr, lambda: base_lr / 2.0)
  return tf_v1.train.AdamOptimizer(lr, 0.5)

