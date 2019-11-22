"""
Eval metrics library
"""

import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2

# Linear composition of an objective function

def discriminator_loss(discriminator, y_synth, alpha):
    input_fn = lambda: y_synth
    return alpha*discriminator.evaluate(input_fn)

def summarizer_loss(y_truth, y_synth, beta):
    return beta*tf_v2.image.ssim(y_synth, y_truth, 255)

def objective(model, discriminator, x, y, alpha, beta):
  y_ = model(x)
  return discriminator_loss(discriminator, y_, alpha) + summarizer_loss(y, y_, beta)

def grad(model, discriminator, inputs, targets, alpha, beta):
  with tf_v1.GradientTape() as tape:
    loss_value = objective(model, discriminator, inputs, targets, alpha, beta)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Other eval metrics.
#TODO: Frechet Inception Distance