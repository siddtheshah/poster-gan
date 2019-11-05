import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from PIL import Image
import time

#TODO: set up argparsing.

RESULTS_PATH = "results"
RUN_NAME = "first"

STEPS_PER_EVAL = 50  # @param
MAX_TRAIN_STEPS = 500  # @param
BATCHES_FOR_EVAL_METRICS = 10  # @param

RUN_FOLDER = os.path.join(os.getcwd(), RESULTS_PATH, RUN_NAME)

logging.basicConfig(filename=os.path.join(RUN_FOLDER, "log.txt"), level=logging.DEBUG)

from tfgan_quick_train_lib import *

gan_estimator = tfgan.estimator.GANEstimator(
    generator_fn=unconditional_generator,
    discriminator_fn=unconditional_discriminator,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    params={'batch_size': train_batch_size, 'noise_dims': noise_dimensions},
    generator_optimizer=gen_opt,
    discriminator_optimizer=tf.train.AdamOptimizer(discriminator_lr, 0.5),
    get_eval_metric_ops_fn=get_eval_metric_ops_fn)


# Used to track metrics.
steps = []
real_logits, fake_logits = [], []
real_mnist_scores, mnist_scores, frechet_distances = [], [], []

cur_step = 0
start_time = time.time()
while cur_step < MAX_TRAIN_STEPS:
    next_step = min(cur_step + STEPS_PER_EVAL, MAX_TRAIN_STEPS)

    start = time.time()
    gan_estimator.train(input_fn, max_steps=next_step)
    steps_taken = next_step - cur_step
    time_taken = time.time() - start
    print('Time since start: %.2f min' % ((time.time() - start_time) / 60.0))
    print('Trained from step %i to %i in %.2f steps / sec' % (
        cur_step, next_step, steps_taken / time_taken))
    cur_step = next_step

    # Calculate some metrics.
    metrics = gan_estimator.evaluate(input_fn, steps=BATCHES_FOR_EVAL_METRICS)
    steps.append(cur_step)
    real_logits.append(metrics['real_data_logits'])
    fake_logits.append(metrics['gen_data_logits'])

    print('Average discriminator output on Real: %.2f  Fake: %.2f' % (
        real_logits[-1], fake_logits[-1]))
    print('Inception Score: %.2f / %.2f  Frechet Distance: %.2f' % (
        mnist_scores[-1], real_mnist_scores[-1], frechet_distances[-1]))

    # Visualize some images.
    iterator = gan_estimator.predict(
        input_fn, hooks=[tf.train.StopAtStepHook(num_steps=21)])

    try:
        imgs = np.array([iterator.__next__() for _ in range(20)])
    except StopIteration:
        pass

    tiled = tfgan.eval.python_image_grid(imgs, grid_shape=(2, 10))
    im = Image.fromarray(tiled)
    im.save(os.path.join(RUN_FOLDER, "iter" + cur_step + ".png"))


#TODO: Print figures in run, and save them to files.
plt.title('MNIST Frechet distance per step')
plt.plot(steps, frechet_distances)
plt.figure()
plt.title('MNIST Score per step')
plt.plot(steps, mnist_scores)
plt.plot(steps, real_mnist_scores)
plt.show()