"""
Eval metrics library
"""
import summarizer.dataset

import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2
import os
import matplotlib.pyplot as plt
import numpy as np

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

#########################################################
# Image plotting function
#########################################################

def show_predict_comparison(model, eval_ids, results_dir, poster_dir, trailer_dir):
    plt.figure()
    rows = []
    for id in eval_ids:
        trailer_frames, poster = summarizer.train.make_summary_example(id, poster_dir, trailer_dir)
        prediction = model.predict(trailer_frames)
        rows.append(np.hstack(poster, prediction))

    concat = np.vstack(rows)
    plt.imshow(concat)
    plt.savefig(os.path.join(results_dir + "predictions.png"))