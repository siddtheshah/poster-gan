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
def discriminator_loss(alpha, discriminator, generator):
    # with summary_graph.as_default() as graph:
    def loss(summarizer_output, y_truth):
        with summary_graph.as_default():
            y_synth = generator(summarizer_output)["output_1"]
            fake_score = discriminator(y_synth)["output_1"]
            return alpha * (1 - fake_score)
    return loss

# Summarizer loss is the summarizer's ability to generate the poster that corresponds to the trailer.
# It is weighted by beta.
def summarizer_loss(beta, generator):
    def loss(summarizer_output, y_truth):
        with summary_graph.as_default():
            # print(generator.graph)
            # print(summarizer_output.graph)
            y_synth = generator(summarizer_output)["output_1"]
            return beta * tf_v2.image.ssim(y_synth, y_truth, 255)
    return loss

def color_loss(gamma, bins):
    def hists(img):
        hist = np.histogramdd(np.reshape(img, (-1, 3)), bins=bins, density=True)[0]
        hist = hist.astype(np.float32)
        # We can flatten the multi-dim array because KL Div doesn't care about neighborhoods.
        return hist.flatten()

    def loss(summarizer_output, y_truth):
        with summary_graph.as_default():
            hist1 = tf_v1.numpy_function(hists, [summarizer_output], tf_v1.float32)
            hist2 = tf_v1.numpy_function(hists, [y_truth], tf_v1.float32)
            return gamma*(tf_v1.reduce_sum(((hist1-hist2)**2)))
    return loss

# Combination of both summarizer loss and discriminator loss
def combined_loss(alpha, beta, gamma, generator, discriminator, bins):
    def hists(img):
        hist = np.histogramdd(np.reshape(img, (-1, 3)), bins=bins, density=True)[0]
        hist = hist.astype(np.float32)
        # We can flatten the multi-dim array because KL Div doesn't care about neighborhoods.
        return hist.flatten()

    def loss(summarizer_output, y_truth):
        with summary_graph.as_default():
            y_synth = generator(summarizer_output)["output_1"]
            fake_score = discriminator(y_synth)["output_1"]

            hist1 = tf_v1.numpy_function(hists, [summarizer_output], tf_v1.float32)
            hist2 = tf_v1.numpy_function(hists, [y_truth], tf_v1.float32)
            color_loss = gamma*(tf_v1.reduce_sum(((hist1-hist2)**2)))
            summarizer_loss = beta * tf_v2.image.ssim(y_synth, y_truth, 255)
            discriminator_loss = alpha*(1 - fake_score)
            return  summarizer_loss + discriminator_loss + color_loss

    return loss



# Other eval metrics.
# TODO: Frechet Inception Distance

#########################################################
# Eval functions
#########################################################

def show_poster_predict_comparison(sm, generator, results_dir, trailer_dir, poster_dir):
    eval_ids = summarizer.dataset.get_useable_ids(trailer_dir, poster_dir)[:10]
    plt.figure()
    rows = []
    for id in eval_ids:
        trailer_frames, poster = summarizer.dataset.make_summary_example(str(id), poster_dir, trailer_dir)
        stacked_frames = tf_v1.stack(trailer_frames, 0)
        batched = tf_v1.reshape(stacked_frames, [1, 20, 64, 64, 3])
        summary = sm(batched)
        generated_poster = generator(summary)
        single_prediction = tf_v1.reshape(generated_poster, [64, 64, 3])
        rows.append(np.hstack([poster, single_prediction]))

    concat = np.vstack(rows)
    plt.imshow(concat)
    plt.savefig(os.path.join(results_dir, "predictions.png"))
