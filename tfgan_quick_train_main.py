from tfgan_quick_train_lib import *
import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2
import tensorflow_gan as tfgan
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import time

#TODO: set up argparsing.

RUN_NAME="11_10_2019_1"

# Fully qualify these paths when on cloud, because our data will be on disk


IMAGE_DIR = "/mnt/disks/new_space/movie_poster_images"
RUN_DIR = os.path.join("/mnt/disks/new_space/results", RUN_NAME)
MODEL_DIR = os.path.join("/mnt/disks/new_space/model", RUN_NAME)

STEPS_PER_EVAL = 50  # @param
MAX_TRAIN_STEPS = 500  # @param
BATCHES_FOR_EVAL_METRICS = 10  # @param

if not os.path.exists(RUN_DIR):
    os.makedirs(RUN_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

logging.basicConfig(filename=os.path.join(RUN_DIR, "log.txt"), level=logging.DEBUG)

def input_fn(mode, params, shuffle_control=False):
    assert 'batch_size' in params
    assert 'noise_dims' in params
    bs = params['batch_size']
    nd = params['noise_dims']
    split = 'train' if mode == tf_v1.estimator.ModeKeys.TRAIN else 'test'
    shuffle = shuffle_control and (mode == tf_v1.estimator.ModeKeys.TRAIN)
    just_noise = (mode == tf_v1.estimator.ModeKeys.PREDICT)

    noise_ds = (tf_v1.data.Dataset.from_tensors(0).repeat()
                .map(lambda _: tf_v1.random_normal([bs, nd])))

    if just_noise:
        return noise_ds

    image_names = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    dataset = tf_v1.data.Dataset.from_tensor_slices((image_names))

    def _add_images(filename):
        image_string = tf_v1.read_file(filename)
        image_decoded = tf_v1.image.decode_jpeg(image_string, channels=3)
        image_decoded = tf_v1.image.resize_image_with_crop_or_pad(image_decoded, 256, 256)
        image_decoded = tf_v2.expand_dims(image_decoded, 0)
        image = tf_v1.cast(image_decoded, tf_v1.float32)

        print("Input image shape:", image.get_shape(), flush=True)

        # Avg pooling image to reduce resolution.
        # 4x4 kernel

        # image = tf_v1.image.pool(image, window_shape=[4, 4], strides=[4, 4], pooling_type="AVG", padding='SAM
        image = tf_v1.image.resize(image, [64, 64])

        # 64 x 64 image with 3 channels
        image = tf_v1.reshape(image, [64,64,3])
        print("Input image shape:", image.get_shape(), flush=True)
        image = (image - 127.5) / 127.5
        arr = tf_v1.cast(image, tf_v1.float32)
        return arr

    images_ds = dataset.map(_add_images).cache().repeat()
    if shuffle:
        images_ds = images_ds.shuffle(
            buffer_size=10000, reshuffle_each_iteration=True)
    images_ds = (images_ds.batch(bs, drop_remainder=True)
                 .prefetch(tf_v1.data.experimental.AUTOTUNE))

    return tf_v1.data.Dataset.zip((noise_ds, images_ds))

print("Creating estimator")

gan_estimator = tfgan.estimator.GANEstimator(
    model_dir=MODEL_DIR,
    generator_fn=unconditional_generator,
    discriminator_fn=unconditional_discriminator,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    params={'batch_size': train_batch_size, 'noise_dims': noise_dimensions},
    generator_optimizer=gen_opt,
    discriminator_optimizer=tf_v1.train.AdamOptimizer(discriminator_lr, 0.5),
    get_eval_metric_ops_fn=get_eval_metric_ops_fn)

print("Done creating estimator", flush=True)

# Used to track metrics.
steps = []
real_logits, fake_logits = [], []
real_mnist_scores, mnist_scores, frechet_distances = [], [], []

cur_step = 0
start_time = time.time()
while cur_step < MAX_TRAIN_STEPS:
    next_step = min(cur_step + STEPS_PER_EVAL, MAX_TRAIN_STEPS)

    start = time.time()
    print("Training instance")
    gan_estimator.train(input_fn, max_steps=next_step)
    print("Done training on instance")
    steps_taken = next_step - cur_step
    time_taken = time.time() - start
    print('Time since start: %.2f min' % ((time.time() - start_time) / 60.0))
    print('Trained from step %i to %i in %.2f steps / sec' % (
        cur_step, next_step, steps_taken / (time_taken + 1e-10)))
    cur_step = next_step

    # Calculate some metrics.

    metrics = gan_estimator.evaluate(input_fn, steps=BATCHES_FOR_EVAL_METRICS)
    steps.append(cur_step)
    real_logits.append(metrics['real_data_logits'])
    fake_logits.append(metrics['gen_data_logits'])
    real_id.append(metrics['real_data_id'])
    fake_id.append(metrics['fake_data_id'])

    print('Average discriminator output on Real: %.2f  Fake: %.2f' % (
        real_logits[-1], fake_logits[-1]))
    # print('Inception Score: %.2f / %.2f  Frechet Distance: %.2f' % (
    #     mnist_scores[-1], real_mnist_scores[-1], frechet_distances[-1]))

    # Visualize some images.
    iterator = gan_estimator.predict(
        input_fn, hooks=[tf_v1.train.StopAtStepHook(num_steps=21)])


    try:
        imgs = [iterator.__next__() for _ in range(20)]
    except StopIteration:
        pass
    row1 = np.hstack(imgs[:5])
    row2 = np.hstack(imgs[5:10])
    row3 = np.hstack(imgs[10:15])
    row4 = np.hstack(imgs[15:])
    all_images = np.vstack((row1, row2, row3, row4))
    all_images = (all_images + 1.0)*127.5
    plt.imshow(all_images)
    plt.savefig(os.path.join(RUN_DIR, "iter" + str(cur_step) + ".png"))

    plt.figure()
    plt.title('Training plot')
    plt.plot(steps, real_data_id)
    plt.plot(steps, fake_data_id)
    plt.savefig(os.path.join(RUN_DIR, "training_plot.png"))
