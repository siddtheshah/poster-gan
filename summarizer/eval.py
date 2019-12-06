"""
Eval metrics library
"""

import summarizer.dataset
import summarizer.graph

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

def summarizer_loss(beta, generator):
    def loss(summarizer_output, y_truth):
        with summary_graph.as_default():
            linearized = tf_v1.reshape(summarizer_output, [-1])
            slice = tf_v1.expand_dims(tf_v1.slice(linearized, [0], [100]), 0)

            # print(generator.graph)
            # print(summarizer_output.graph)
            y_synth = generator(slice)["out"]
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

def combined_loss(alpha, beta, gamma, generator, discriminator, bins):
    def hists(img):
        img_range = np.array([[0, 256], [0, 256], [0, 256]], dtype='float')
        hist = np.histogramdd(np.reshape(img, (-1, 3)), bins=bins, density=True, range=img_range)[0]
        hist = hist.astype(np.float32)*bins**3
        # We can flatten the multi-dim array because KL Div doesn't care about neighborhoods.
        return hist.flatten()

    def loss(summarizer_output, y_truth):
        with summary_graph.as_default():

            linearized = tf_v1.reshape(summarizer_output, [-1])
            slice = tf_v1.expand_dims(tf_v1.slice(linearized, [0], [100]), 0)
            pad = tf_v1.pad(slice, [[0, 59], [0, 0]])
            y_synth = generator(pad)["out"]
            y_synth_resize = tf_v1.image.resize(y_synth, (64, 64)) 
            
            y_synth_single = y_synth_resize[0, :, :, :]
            y_truth = tf_v1.squeeze(y_truth[0, :, :, :])
            
            print(y_synth.shape)
            print(y_truth.shape)
            try:
                fake_score = discriminator(y_synth)["out"]
                fake_score_single = fake_score[0]
            except:
                fake_score_single = 1

            hist1 = tf_v1.numpy_function(hists, [y_synth_single], tf_v1.float32)
            hist2 = tf_v1.numpy_function(hists, [y_truth], tf_v1.float32)
            color_loss = gamma*(tf_v1.reduce_sum(((hist1-hist2)**2)))
            summarizer_loss = beta * (1 - tf_v2.image.ssim(y_synth_single, y_truth, 255))
            discriminator_loss = alpha*(1 - fake_score_single)
            return summarizer_loss + discriminator_loss + color_loss

    return loss
# Other eval metrics.
# TODO: Frechet Inception Distance

#########################################################
# Eval functions
#########################################################

def show_training_plot(history, results_dir):
    plt.plot(history.history['loss_1'])
    plt.plot(history.history['loss_2'])
    plt.plot(history.history['val_loss_1'])
    plt.plot(history.history['val_loss_2'])
    plt.title('Loss during Training')
    plt.ylabel('Weighted Loss')
    plt.xlabel('Epoch')
    plt.legend(['SSIM Train Loss', 'Color Train Loss', 'SSIM Validation Loss', 'Color Validation Loss'], loc='upper left')
    plt.savefig(os.path.join(results_dir, "training_plot.png"))

def show_poster_mock_predict_comparison(sm, generator, results_dir, trailer_dir, poster_dir):
    eval_ids = summarizer.dataset.get_useable_ids(trailer_dir, poster_dir)[:10]
    print("Running Eval on ", eval_ids)
    rows = []
    for id in eval_ids:
        trailer_frames, poster = summarizer.dataset.make_summary_example(str(id), poster_dir, trailer_dir)
        poster = poster*255
        stacked_frames = tf_v1.stack(trailer_frames, 0)
        batched = tf_v1.reshape(stacked_frames, [1, 20, 64, 64, 3])
        summary = sm(batched)
        # print(summary)
        linearized = tf_v1.reshape(summary, [-1])
        slice = tf_v1.expand_dims(tf_v1.slice(linearized, [1000], [1100]), 0)
        generated_poster = generator(summary)
        single_prediction = tf_v1.reshape(generated_poster, [64, 64, 3])
        rows.append(np.hstack([poster, single_prediction]))

    concat = np.vstack(rows).astype(int)
    plt.figure()
    plt.imshow(concat)
    plt.savefig(os.path.join(results_dir, "predictions.png"))
