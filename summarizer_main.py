import summarizer
import summarizer.graph
import summarizer.mock_gan
import summarizer.eval
from summarizer.network import SummaryNetwork

import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2
import numpy as np

import os
import json
import argparse

# Make sure you set all the extra options in summarizer_config.json before running this script
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train', default=False, action='store_true', help='Train a new model')
parser.add_argument('--eval', default=False, action='store_true', help='Run eval on the model')
parser.add_argument('--run_name', help='Specify a run name. (Required)', action='store')
args = parser.parse_args()

global summary_graph
summary_graph = summarizer.graph.summary_graph

def train_new_model(configs):

    # Compose the training step
    generator = tf_v2.saved_model.load(export_dir=configs["generator_path"])
    generator_predict = generator.signatures["serving_default"]
    discriminator = tf_v2.saved_model.load(export_dir=configs["discriminator_path"])
    discriminator_predict = discriminator.signatures["serving_default"]

    alpha = configs["alpha"]
    beta = configs["beta"]
    batch_size = configs["batch_size"]

    run_dir = os.path.join(configs["storage_dir"], args.run_name)
    save_dir = os.path.join(run_dir, "model")

    with summary_graph.as_default():
        with tf_v1.Session(graph=summary_graph) as sess:
            train_dataset, validation_dataset = \
                summarizer.dataset.create_summary_dataset(configs["trailer_dir"], configs["poster_dir"], batch_size)
            model = SummaryNetwork(configs["weight_decay"])
            summarizer_loss = summarizer.eval.summarizer_loss(beta, generator_predict)
            discriminator_loss = summarizer.eval.discriminator_loss(alpha, discriminator_predict, generator_predict)
            combined_loss = summarizer.eval.combined_loss(alpha, beta, generator_predict, discriminator_predict)
            model.compile(optimizer=tf_v1.keras.optimizers.Adam(), loss=combined_loss,
                          metrics=[summarizer_loss, discriminator_loss])

            # model.compile(optimizer=tf_v1.keras.optimizers.RMSprop(), loss=discriminator_loss, metrics=[])
            # model.compile(optimizer=tf_v1.keras.optimizers.RMSprop(), loss=tf_v1.keras.losses.mean_squared_error, metrics=[])
            checkpoint = tf_v1.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, "model"), monitor='val_loss',
                                                               save_best_only=True, verbose=1, mode='min')
            callbacks_list = [checkpoint]
            model.fit(train_dataset, epochs=configs["epochs"], verbose=2,
                      validation_data=validation_dataset, callbacks=callbacks_list)
            model.save(os.path.join(configs["storage_dir"], args.run_name, "model"), save_format='tf')
            print("Model finished training.")

def eval_model(configs):
    run_dir = os.path.join(configs["storage_dir"], args.run_name)
    results_dir = os.path.join(run_dir, "results")
    save_dir = os.path.join(run_dir, "model")
    model = tf_v2.saved_model.load(save_dir)

    generator = tf_v2.saved_model.load(export_dir=os.path.join(configs["storage_dir"], "generator"))

    # Show real poster vs predictions
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    summarizer.eval.show_poster_predict_comparison(model, generator, configs["eval_ids"], results_dir, configs["poster_dir"],
                                                   configs["trailer_dir"])

    # Other evaluation metrics


def main():
    with open('summarizer_config.json') as config_file:
        configs = json.load(config_file)
    if args.train:
        train_new_model(configs)
    if args.eval:
        eval_model(configs)


if __name__ == "__main__":
    main()
