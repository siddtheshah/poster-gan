import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v1 as tf_v2

class SummaryNetwork(tf_v1.keras.Model):
    def __init__(self, generator, weight_decay):
        super(SummaryNetwork, self).__init__()
        # Define network layers
        self.generator_ = generator


    def call(self, inputs):
        # Pass inputs to layers
        pass

