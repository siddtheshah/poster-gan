import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2

class SummaryTrainer:
    """
    A class that takes in tensorflow components and ex
    """
    def __init__(self, model, dataset, storage_dir, optimizer, loss, batch_size=32, steps_per_validation=100):
        _optimizer = optimizer
        _batch_size = batch_size
        _dataset = dataset         # A SummaryDataset, not a tf.data.Dataset!
        _storage_dir = storage_dir
        _model = model
        _loss = loss
        _steps_per_validation = steps_per_validation

    def train(self, training_steps):
        """
        Executes the training loop.
        Args:
         training_steps: Number of steps to train for
        """
        for i in range(training_steps):


            if i % self.steps_per_validation == 0:
                self.validate()


    def validate(self):
        pass


