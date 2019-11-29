import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf_v2


# This mock generator just returns the input as its output.

# When the summarization model is using this as it's generator, it
# is trying to produce a poster without understanding what a poster is.
class MockGenerator(tf_v1.keras.Model):
    def __init__(self):
        super(MockGenerator, self).__init__()
        # self.cn1 = tf_v1.keras.layers.Conv2D(3, (3,3), padding='same', activation='relu', input_shape=(64, 64, 3))

    def call(self, inputs):
        # outputs = self.cn1(inputs)
        return inputs


# This mock discriminator just returns 0 as its output.

# When the summarization model tries to use this for feedback, it gets nothing,
# and it doesn't get a clue about what a poster is.
class MockDiscriminator(tf_v1.keras.Model):
    def __init__(self):
        super(MockDiscriminator, self).__init__()

    def call(self, inputs):
        return tf_v2.constant(0.0)
