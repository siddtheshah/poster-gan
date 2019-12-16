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
parser.add_argument('--mock', help='Run the summarizer with a mock generator/discriminator', action='store_true',
                    default=False)
args = parser.parse_args()

global summary_graph
summary_graph = summarizer.graph.summary_graph

# tf_v1.enable_eager_execution()

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

    discriminator_path = configs["discriminator_path"]
    generator_path = configs["generator_path"]

    if args.mock:
        generator_path = os.path.join(generator_path, "mock")
        discriminator_path = os.path.join(discriminator_path, "mock")

        generator = MockGenerator()
        generator._set_inputs(tf_v1.keras.Input(shape=(64, 64, 3)))
        generator.compile(loss='mse', optimizer='rmsprop')
        generator.save(generator_path, save_format='tf')
        discriminator = MockDiscriminator()
        discriminator._set_inputs(tf_v1.keras.Input(shape=(64, 64, 3)))
#        discriminator.compile(loss='mse', optimizer='rmsprop')
        discriminator.save(discriminator_path, save_format='tf')



    alpha = configs["alpha"]
    beta = configs["beta"]
    gamma = configs["gamma"]

    run_dir = os.path.join(configs["storage_dir"], args.run_name)
    save_dir = os.path.join(run_dir, "model")
    results_dir = os.path.join(run_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    generator = tf_v2.saved_model.load(export_dir=generator_path)
    generator_predict = generator.signatures["serving_default"]

    with summary_graph.as_default():
        with tf_v1.Session(graph=summary_graph) as sess:
            # Compose the training step

            discriminator = tf_v2.saved_model.load(export_dir=discriminator_path)
            discriminator_predict = discriminator.signatures["serving_default"]

            init = tf_v1.global_variables_initializer()
            sess.run(init) # Initialize the models loaded above.

            if args.mock:
                summarizer_model = SummaryNetwork(configs["weight_decay"], configs["batch_size"])
            else:
                summarizer_model = SummaryNetwork(configs["weight_decay"], configs["batch_size"], z_dim=configs["noise_dim"])

            train_dataset, validation_dataset = \
                summarizer.dataset.create_summary_dataset(configs["trailer_dir"], configs["poster_dir"],
                                                          configs["batch_size"])
            if args.mock:
                summarizer_loss = summarizer.eval.summarizer_loss_mock(beta, generator_predict)
                discriminator_loss = summarizer.eval.discriminator_loss_mock(alpha, discriminator_predict, generator_predict)
                combined_loss = summarizer.eval.combined_loss_mock(alpha, beta, gamma, generator_predict,
                                                              discriminator_predict, 12)
                cap = summarizer.network.make_mock_cap()
            else:
                summarizer_loss = summarizer.eval.summarizer_loss(beta)
                combined_loss = summarizer.eval.combined_loss(alpha, beta, gamma, discriminator_predict, 12)
                cap = summarizer.network.make_gan_cap(generator_predict)

            input = tf_v1.keras.layers.Input(shape=(20, 64, 64, 3))
            sm_output = summarizer_model(input)
            output = cap(sm_output)

            model = tf_v1.keras.Model(inputs=input, outputs=output)

            color_loss = summarizer.eval.color_loss(gamma, 12)
            model.compile(optimizer=tf_v1.keras.optimizers.Adam(learning_rate=configs["learning_rate"]),
                          loss=combined_loss, metrics=[summarizer_loss, color_loss],
                          experimental_run_tf_function=False)
            checkpoint = tf_v1.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, "model"), monitor='val_loss',
                                                               save_best_only=True, verbose=1, mode='min')
            callbacks_list = [] #[checkpoint]
            training_history = model.fit(train_dataset, epochs=configs["epochs"], verbose=2,
                       validation_data=validation_dataset, callbacks=callbacks_list)
            summarizer_model.save(os.path.join(configs["storage_dir"], args.run_name, "model"), save_format='tf')
            summarizer.eval.show_training_plot(training_history, results_dir)
            print("Model finished training.")


def eval_model(configs):
    run_dir = os.path.join(configs["storage_dir"], args.run_name)
    results_dir = os.path.join(run_dir, "results")
    save_dir = os.path.join(run_dir, "model")

    # Show real poster vs predictions
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    generator_path = configs["generator_path"]
    if args.mock:
        generator_path = os.path.join(generator_path, "mock")
    generator = tf_v2.saved_model.load(export_dir=generator_path)
    generator_predict = generator.signatures["serving_default"]

    summarizer_model = tf_v2.saved_model.load(save_dir)
    if args.mock:
        cap = summarizer.network.make_mock_cap()
    else:
        cap = summarizer.network.make_gan_cap(generator_predict)

    summarizer_predict = summarizer_model.signatures["serving_default"]
    summarizer.eval.show_poster_predict_comparison(summarizer_predict, cap , results_dir, configs["trailer_dir"], configs["poster_dir"])
    # Other evaluation metrics

def main():
    with open('config.json') as config_file:
        configs = json.load(config_file)
        # test_gan_io_2(configs)

    if args.train:
        train_new_model(configs)

    if args.eval:
        eval_model(configs)

if __name__ == "__main__":
    main()
