"""
Eval metrics library
"""

import summarizer.dataset
import summarizer.graph
import summarizer.wasserstein

import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2
import os
import matplotlib.pyplot as plt
import numpy as np

tf_v1.enable_eager_execution()
global summary_graph
summary_graph = summarizer.graph.summary_graph


##################################################
# Loss functions
##################################################

# Discriminator loss is the discriminator's belief that the generated image is a poster.
# It is weighted by alpha.
def discriminator_loss_mock(alpha, discriminator, generator):
    # with summary_graph.as_default() as graph:
    def loss(summarizer_output, y_truth):
        with summary_graph.as_default():
            y_synth = generator(summarizer_output)["output_1"]
            fake_score = discriminator(y_synth)["output_1"]
            return alpha * (1 - fake_score)
    return loss

# Summarizer loss is the summarizer's ability to generate the poster that corresponds to the trailer.
# It is weighted by beta.
def summarizer_loss_mock(beta, generator):
    def loss(summarizer_output, y_truth):
        with summary_graph.as_default():
            # print(generator.graph)
            # print(summarizer_output.graph)
            y_synth = generator(summarizer_output)["output_1"]
            return beta * (1 - tf_v2.image.ssim(y_synth, y_truth, 255))
    return loss

def summarizer_loss(beta):
    def loss(y_synth, y_truth):
        y_truth_compact = tf_v1.image.resize(y_truth, (64, 64))
        return beta * (1 - tf_v2.image.ssim(y_synth, y_truth_compact, 255))
    return loss

def color_loss(gamma, bins):
    def hists(img):
        img_range = np.array([[0, 256], [0, 256], [0, 256]], dtype='float')
        hist = np.histogramdd(np.reshape(img, (-1, 3)), bins=bins, density=True, range=img_range)[0]
        hist = hist.astype(np.float32)*bins**3  # Denormalize by bin count.
        # We can flatten the multi-dim array because KL Div doesn't care about neighborhoods.
        return hist.flatten()

    def loss(summarizer_output, y_truth):
        with summary_graph.as_default():
            hist1 = tf_v1.numpy_function(hists, [summarizer_output], tf_v1.float32)
            hist2 = tf_v1.numpy_function(hists, [y_truth], tf_v1.float32)
            return gamma*(tf_v1.reduce_sum(((hist1-hist2)**2)))
    return loss

# Combination of both summarizer loss and discriminator loss
def combined_loss_mock(alpha, beta, gamma, generator, discriminator, bins):
    def hists(img):
        img_range = np.array([[0, 256], [0, 256], [0, 256]], dtype='float')
        hist = np.histogramdd(np.reshape(img, (-1, 3)), bins=bins, density=True, range=img_range)[0]
        hist = hist.astype(np.float32)*bins**3
        # We can flatten the multi-dim array because KL Div doesn't care about neighborhoods.
        return hist.flatten()

    def loss(summarizer_output, y_truth):
        with summary_graph.as_default():

            y_synth = generator(summarizer_output)["output_1"]
            fake_score = discriminator(y_synth)["output_1"]

            hist1 = tf_v1.numpy_function(hists, [summarizer_output], tf_v1.float32)
            hist2 = tf_v1.numpy_function(hists, [y_truth], tf_v1.float32)
            color_loss = gamma*(tf_v1.reduce_sum(((hist1-hist2)**2)))
            summarizer_loss = beta * (1 - tf_v2.image.ssim(y_synth, y_truth, 255))
            discriminator_loss = alpha*(1 - fake_score)
            return summarizer_loss + discriminator_loss + color_loss

    return loss

def combined_loss(alpha, beta, gamma, delta, discriminator, bins):
    def hists(img):
        img = img.astype(int)
        #print(img)
        img_range = np.array([[0, 256], [0, 256], [0, 256]], dtype='float')
        hist = np.histogramdd(np.reshape(img, (-1, 3)), bins=bins, density=True, range=img_range)[0]
        hist = hist.astype(np.float32) * bins ** 3
        # We can flatten the multi-dim array because KL Div doesn't care about neighborhoods.
        return hist.flatten()

    def loss(y_synth, y_truth):
        y_synth = tf_v1.image.resize(y_synth, (64, 64))
        y_truth = (y_truth + 1.0)*127.5
        y_synth_full = tf_v1.image.resize(y_synth, (256, 256))

        y_truth = tf_v1.image.resize(y_truth, (64, 64))
        fake_score = discriminator(y_synth_full)["discriminator_output"]

        hist1 = tf_v1.numpy_function(hists, [y_synth], tf_v1.float32)
        hist2 = tf_v1.numpy_function(hists, [y_truth], tf_v1.float32)
        wasserstein = summarizer.wasserstein.Wasserstein(1, hist1, hist2)
        color_loss = gamma*wasserstein.dist(C=.1, nsteps=10, reset=True)
        # color_loss = gamma * (tf_v1.reduce_sum(((hist1 - hist2) ** 2)))

        summarizer_loss = beta * (1 - tf_v2.image.ssim(y_synth, y_truth, 255))
        discriminator_loss = alpha * (1 - tf_v1.nn.sigmoid(fake_score))

        mse_loss = tf_v1.losses.mean_squared_error(y_synth, y_truth)
        nmse_loss = delta * mse_loss/(tf_v1.math.reduce_mean(y_synth) * tf_v1.math.reduce_mean(y_truth))
        return (summarizer_loss + discriminator_loss + color_loss + nmse_loss)/(alpha + beta + gamma + delta)

    return loss
# Other eval metrics.
# TODO: Frechet Inception Distance

#########################################################
# Eval functions
#########################################################

def show_training_plot(history, results_dir):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss during Training')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
    plt.savefig(os.path.join(results_dir, "training_plot.png"))

def show_poster_predict_comparison(summarizer_predict, cap, results_dir, trailer_dir, poster_dir):
    eval_ids = summarizer.dataset.get_useable_ids(trailer_dir, poster_dir)[15:20]
    eval_ids = [114709, 111161, 68646, 120737, 1375666]
    print("Running Eval on ", eval_ids)
    rows = []
    for i, id in enumerate(eval_ids):
        trailer_frames, poster = summarizer.dataset.make_summary_example(str(id), poster_dir, trailer_dir)
        # poster = poster*255
        stacked_frames = tf_v1.stack(trailer_frames, 0)
        single_batch = tf_v1.expand_dims(stacked_frames, 0)

        sm_output = summarizer_predict(single_batch)["output_1"]
        # sm_output = tf_v1.random.uniform((1, 1024), minval=-.5, maxval=.5)

        output = cap(sm_output)

        y_synth = output

        # y_synth = (y_synth + 1)*127.5
        y_synth = (y_synth + 1)/2

        prediction = tf_v1.squeeze(y_synth)
        # print(prediction)
        rows.append(np.hstack([poster, prediction]))

    concat = np.vstack(rows) #.astype(int)
    plt.figure()
    plt.imshow(concat)
    plt.savefig(os.path.join(results_dir, "predictions.png"))

def show_batched_poster_predict_comparison(summarizer_predict, cap, results_dir, trailer_dir, poster_dir, batch_size, override_ids=None):
    if override_ids:
        if len(override_ids) > batch_size:
            print("Too many ids specified given batch size. Aborting image generation.")
            return
        eval_ids = override_ids + summarizer.dataset.get_useable_ids(trailer_dir, poster_dir)[:batch_size - len(override_ids)]
    else:
        eval_ids = summarizer.dataset.get_useable_ids(trailer_dir, poster_dir)[:batch_size]
    print("Running Eval on ", eval_ids)
    rows = []
    stacked_frames_batch = []
    posters = []
    for i, id in enumerate(eval_ids):
        trailer_frames, poster = summarizer.dataset.make_summary_example(str(id), poster_dir, trailer_dir)
        stacked_frames = tf_v1.stack(trailer_frames, 0)
        stacked_frames_batch.append(stacked_frames)
        posters.append(poster)

    batch = tf_v1.stack(stacked_frames_batch)

    sm_output = summarizer_predict(batch)["output_1"]
        # sm_output = tf_v1.random.uniform((1, 1024), minval=-.5, maxval=.5)

    output = cap(sm_output)
    y_synth = output
    y_synth = (y_synth + 1)/2

    for i in range(batch_size):
        rows.append(np.hstack([posters[i], y_synth[i]]))

    concat = np.vstack(rows) #.astype(int)
    plt.figure()
    plt.imshow(concat)
    plt.savefig(os.path.join(results_dir, "predictions.png"))

def debug_gan_show(cap, results_dir, trailer_dir, poster_dir):
    eval_ids = summarizer.dataset.get_useable_ids(trailer_dir, poster_dir)[:5]
    print("Running Eval on ", eval_ids)
    rows = []
    for i, id in enumerate(eval_ids):
        trailer_frames, poster = summarizer.dataset.make_summary_example(str(id), poster_dir, trailer_dir)
        # poster = poster*255
        stacked_frames = tf_v1.stack(trailer_frames, 0)
        single_batch = tf_v1.expand_dims(stacked_frames, 0)

        # sm_output = summarizer_predict(single_batch)["output_1"]
        sm_output = tf_v1.random.uniform((1, 1024), minval=-.5, maxval=.5)

        output = cap(sm_output)

        y_synth = output

        # y_synth = (y_synth + 1)*127.5
        y_synth = (y_synth + 1)/2

        # y_synth_resize = tf_v1.image.resize(y_synth, (64, 64))

        prediction = y_synth
        rows.append(np.hstack([prediction[i] for i in range(prediction.shape[0])]))

    concat = np.vstack(rows) #.astype(int)
    plt.figure()
    plt.imshow(concat)
    plt.savefig(os.path.join(results_dir, "debug.png"))