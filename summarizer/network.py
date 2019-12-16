import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v1 as tf_v2
import summarizer.graph

global summary_graph
summary_graph = summarizer.graph.summary_graph

# tf_v1.disable_eager_execution()

def make_mock_cap():

    pad = tf_v1.keras.layers.ZeroPadding2D(padding=(1, 1))
    def mock_cap(x):

        x = 127.5 * (x + 1)  # Get back on 0-255 scale
        x = tf_v1.reshape(x, [-1, 62, 62, 3])  # Keras Conv3D doesn't allow directional 'same' padding for kernels
        # x = tf_v1.Print(x, [x])

        x = pad(x)
        return x
    return mock_cap

def make_gan_cap(generator_predict):
    def call_generator(input):
        # return tf_v1.zeros([input.shape[0], 64, 64, 3])
        return generator_predict(input)["generator_output"]

    def gan_cap(x):
        print(x.shape)
        # x = self.generator_predict(x)["generator_output"]

        x = tf_v1.py_function(call_generator, [x], tf_v1.float32) # This works, but makes model unsaveable
        # x = self.generator_predict(x)["generator_output"]

        x = tf_v1.reshape(x, [-1, 64, 64, 3])
        return x

    return gan_cap

class SummaryNetwork(tf_v1.keras.Model):
    def __init__(self, weight_decay, batch_size, filters=40, z_dim=None):
        super(SummaryNetwork, self).__init__()
        self.batch_size = batch_size
        self.z_dim = z_dim
        # Define network layers
        # Using Keras's ConvLSTM2D to utilize the temporal dimension of a trailer in a basic way.

        self.cl_1 = tf_v1.keras.layers.ConvLSTM2D(filters=filters, strides=1, kernel_size=(3, 3),
                                                  padding='same', return_sequences=True,
                                                  # kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
                                                  # bias_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
                                                  kernel_initializer='random_uniform',
                                                  bias_initializer='zeros')
        self.bn_1 = tf_v1.keras.layers.BatchNormalization()
        self.cl_2 = tf_v1.keras.layers.ConvLSTM2D(filters=filters, strides=1, kernel_size=(3, 3),
                                                  padding='same', return_sequences=True,
                                                  # kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
                                                  # bias_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
                                                  kernel_initializer='random_uniform',
                                                  bias_initializer='zeros')
        self.bn_2 = tf_v1.keras.layers.BatchNormalization()
        self.cl_3 = tf_v1.keras.layers.ConvLSTM2D(filters=filters, strides=1, kernel_size=(3, 3),
                                                  padding='same', return_sequences=True,
                                                  # kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
                                                  # bias_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
                                                  kernel_initializer='random_uniform',
                                                  bias_initializer='zeros')
        self.bn_3 = tf_v1.keras.layers.BatchNormalization()
        self.cl_4 = tf_v1.keras.layers.ConvLSTM2D(filters=filters, strides=1, kernel_size=(3, 3),
                                                  padding='same', return_sequences=True,
                                                  # kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
                                                  # bias_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
                                                  kernel_initializer='random_uniform',
                                                  bias_initializer='zeros')
        self.bn_4 = tf_v1.keras.layers.BatchNormalization()
        # Layer for combined model



        self.pad = tf_v1.keras.layers.ZeroPadding2D(padding=(1, 1))

        self.conv = tf_v1.keras.layers.Conv3D(filters=3, strides=1, kernel_size=(20, 3, 3),
                                              kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
                                              padding='valid', activation='tanh',
                                              bias_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
                                              kernel_initializer='random_uniform',
                                              bias_initializer='zeros')
        if self.z_dim:
            self.ln_1 = tf_v1.keras.layers.Dense(z_dim, activation='tanh')

    def call_generator(self, input):
        # return tf_v1.zeros([input.shape[0], 64, 64, 3])
        return self.generator_predict(input)["generator_output"]

    def call(self, inputs):
        print(inputs.shape)
        x = self.cl_1(inputs)
        x = self.bn_1(x)
        x = self.cl_2(x)
        x = self.bn_2(x)
        x = self.cl_3(x)
        x = self.bn_3(x)
        x = self.cl_4(x)
        x = self.bn_4(x)
        print(x.shape)
        x = self.conv(x)
        print(x.shape)
        if self.z_dim:
            x = tf_v1.reshape(x, [-1, 62 * 62 * 3])
            x = self.ln_1(x)

        return x
