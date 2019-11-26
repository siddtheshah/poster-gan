import summarizer
import summarizer.mock_gan
from summarizer.mock_gan import MockDiscriminator
from summarizer.mock_gan import MockGenerator
import summarizer.eval
import summarizer.train
from summarizer.dataset import SummaryDataset
from summarizer.network import SummaryNetwork

import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2
import numpy as np
import matplotlib.pyplot as plt

import logging
import os
import time
import json
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train', default=False, action='store_true', help='Train a new model')
parser.add_argument('--eval', default=False, action='store_true', help='Run eval on the model')
parser.add_argument('--storage_dir', default='/mnt/disks/new_space/ml/summarizer', action='store', help='Specify path of results and model data')
parser.add_argument('--trailer_dir', default='/mnt/disks/new_space/ml_trailers_key_frame_features', action='store', help='Specify poster data path')
parser.add_argument('--poster_dir', default='/mnt/disks/new_space/movie_poster_images', action='store', help='Specify poster data path')
parser.add_argument('--run_name', help='Specify a run name. (Required)', action='store')
args = parser.parse_args()

tf_v1.enable_eager_execution()

def train_new_model(configs):
    # Create and save a mock generator/discriminator
    generator = MockGenerator()
    generator._set_inputs((64,64,3))
    generator.save(os.path.join(args.storage_dir, "generator"), save_format='tf')
    discriminator = MockDiscriminator()
    discriminator._set_inputs((64, 64, 3))
    discriminator.save(os.path.join(args.storage_dir, "discriminator"), save_format='tf')

    # Compose the training step
    generator = tf_v1.keras.models.load_model(os.path.join(args.storage_dir, "generator"))
    discriminator = tf_v1.keras.models.load_model(os.path.join(args.storage_dir, "discriminator"))

    model = SummaryNetwork(generator, configs["weight_decay"])
    model.compile()
    dataset = SummaryDataset(args.trailer_dir, args.poster_dir, configs["validation_folds"])
    optimizer = tf_v1.keras.optimizers.Adam()
    save_dir = os.path.join(args.storage_dir, args.run_name, "model")
    loss = summarizer.eval.grad
    alpha = configs["alpha"]
    beta = configs["beta"]
    batch_size = configs["batch_size"]
    epochs_per_validation = configs["epochs_per_validation"]
    # Create the trainer and train.
    trainer = summarizer.train.SummaryTrainer(model, discriminator, dataset, save_dir, optimizer, loss, alpha, beta, batch_size, epochs_per_validation)
    trainer.train(configs["epochs"])
    return

def eval_model(configs):
    run_dir = os.path.join(args.storage_dir, args.run_name)
    results_dir = os.path.join(run_dir, "results")
    save_dir = os.path.join(run_dir, "model")
    model = tf_v1.keras.load_model(save_dir)

    # Show real poster vs predictions
    summarizer.eval.show_predict_comparison(model, configs["eval_ids"], results_dir, args.poster_dir, args.trailer_dir)

    # Other evaluation metrics

def main():
    with open(os.path.join('summarizer','config.json')) as config_file:
        configs = json.load(config_file)
    if args.train:
        train_new_model(configs)

    if args.eval:
        eval_model(configs)

if __name__ == "__main__":
    main()
