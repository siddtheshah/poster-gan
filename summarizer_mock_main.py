import summarizer
import summarizer.graph
import summarizer.mock_gan
from summarizer.mock_gan import MockDiscriminator
from summarizer.mock_gan import MockGenerator
import summarizer.eval
from summarizer.network import SummaryNetwork

import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2
import numpy as np

import os
import json
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train', default=False, action='store_true', help='Train a new model')
parser.add_argument('--eval', default=False, action='store_true', help='Run eval on the model')
parser.add_argument('--run_name', help='Specify a run name. (Required)', action='store')
args = parser.parse_args()

global summary_graph
summary_graph = summarizer.graph.summary_graph

def test_gan_io(configs):
    generator = MockGenerator()
    generator._set_inputs(tf_v1.keras.Input(shape=(64.0, 64.0, 3.0)))
    generator.compile(loss='mse', optimizer='rmsprop')
    generator.save(os.path.join(configs["storage_dir"], "generator"), save_format='tf')
    generator = tf_v1.keras.models.load_model(os.path.join(configs["storage_dir"], "generator"))

    discriminator = MockDiscriminator()
    discriminator._set_inputs(tf_v1.keras.Input(shape=(64.0, 64.0, 3.0)))
    discriminator.compile(loss='mse', optimizer='rmsprop')
    discriminator.save(os.path.join(configs["storage_dir"], "discriminator"), save_format='tf')
    discriminator = tf_v1.keras.models.load_model(os.path.join(configs["storage_dir"], "discriminator"))

    z = np.zeros((20, 64, 64, 3))
    z.astype(np.float32)
    res = generator.predict(z, steps=1)
    print(res)
    res = discriminator.predict(z, z, steps=1)
    print(res)

def train_new_model(configs):
    # Create and save a mock generator/discriminator
    generator = MockGenerator()
    generator._set_inputs(tf_v1.keras.Input(shape=(64, 64, 3)))
    generator.compile(loss='mse', optimizer='rmsprop')
    generator.save(os.path.join(configs["storage_dir"], "generator"), save_format='tf')
    discriminator = MockDiscriminator()
    discriminator._set_inputs(tf_v1.keras.Input(shape=(64, 64, 3)))
    discriminator.compile(loss='mse', optimizer='rmsprop')
    discriminator.save(os.path.join(configs["storage_dir"], "discriminator"), save_format='tf')

    # Compose the training step
    generator = tf_v2.saved_model.load(export_dir=os.path.join(configs["storage_dir"], "generator"))
    generator_predict = generator.signatures["serving_default"]
    discriminator = tf_v2.saved_model.load(export_dir=os.path.join(configs["storage_dir"], "discriminator"))
    discriminator_predict = discriminator.signatures["serving_default"]
    print(list(discriminator.signatures.keys()))

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
            model.compile(optimizer=tf_v1.keras.optimizers.Adam(),
                          loss=combined_loss, metrics=[summarizer_loss, discriminator_loss])
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
    summarizer.eval.show_poster_predict_comparison(model, generator, results_dir, configs["trailer_dir"], configs["poster_dir"])

    # Other evaluation metrics

def main():
    with open('summarizer_config.json') as config_file:
        configs = json.load(config_file)
        # test_gan_io_2(configs)

    if args.train:
        train_new_model(configs)

    if args.eval:
        eval_model(configs)

if __name__ == "__main__":
    main()
