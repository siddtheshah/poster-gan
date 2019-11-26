import summarizer.eval

import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2

class SummaryTrainer:
    """
    A class that takes in tensorflow components and ex
    """
    def __init__(self, model, discriminator, dataset, save_dir, optimizer, loss, alpha, beta, batch_size, epochs_per_validation):
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._dataset = dataset         # A SummaryDataset, not a tf.data.Dataset!
        self._save_dir = save_dir
        self._model = model
        self._discriminator = discriminator
        self._loss_fn = loss
        self._alpha = alpha
        self._beta = beta
        self._epochs_per_validation = epochs_per_validation
        self._discriminator_loss = []
        self._summarizer_loss = []
        self._losses = []

    @tf_v1.function
    def train(self, epochs):
        """
        Executes the training loop.
        """
        tf_v1.enable_eager_execution()
        with tf_v1.Session() as sess:
            sess.run(tf_v1.global_variables_initializer())
            for epoch in range(epochs):
                epoch_loss_avg = tf_v1.keras.metrics.Mean()
                train_dataset, validation_dataset = self._dataset.get_split()
                for x, y in train_dataset:
                    # Optimize the model
                    loss_value, grads = self._loss_fn(self._model, self._discriminator, x, y, self._alpha, self._beta)
                    # functools.partial will give parameter binding for above so we don't have to pass discriminator, alpha, or beta
                    self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
                    epoch_loss_avg(loss_value)

                if validation_dataset and epoch % self._epochs_per_validation == 0:
                    print("Validating at Epoch " + str(epoch))
                    self._losses.append(epoch_loss_avg.result())
                    self.validate(validation_dataset)
                    tf_v1.saved_model.save(self._model, self._save_dir)

            tf_v1.saved_model.save(self._model, self._save_dir)
            print("Model finished training.")

    def validate(self, validation_dataset):
        for x, y in validation_dataset:
            y_synth = self._model(x)
            # Optimize the model
            d_loss = summarizer.eval.discriminator_loss(self._discriminator, y_synth, self._alpha)
            s_loss = summarizer.eval.summarizer_loss(y, y_synth, self._beta)
            self._discriminator_loss.append(d_loss)
            self._summarizer_loss.append(s_loss)


