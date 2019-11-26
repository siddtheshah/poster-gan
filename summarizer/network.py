import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v1 as tf_v2

class SummaryNetwork(tf_v1.keras.Model):
    def __init__(self, gan_generator, weight_decay):
        super(SummaryNetwork, self).__init__()
        # Define network layers
        self._generator = gan_generator
        self.cl_1 = tf_v1.keras.layers.ConvLSTM2D(filters=40, strides=1, kernel_size=(3, 3), input_shape=(None, 64, 64, 3),
              padding='same', return_sequences=True, kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
              bias_initializer=tf_v1.keras.regularizers.l2(weight_decay))
        self.bn_1 = tf_v1.keras.layers.BatchNormalization()
        self.cl_2 = tf_v1.keras.layers.ConvLSTM2D(filters=40, strides=1, kernel_size=(3, 3), input_shape=(None, 64, 64, 40),
              padding='same', return_sequences=True, kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
              bias_initializer=tf_v1.keras.regularizers.l2(weight_decay))
        self.bn_2 = tf_v1.keras.layers.BatchNormalization()
        self.cl_3 = tf_v1.keras.layers.ConvLSTM2D(filters=40, strides=1, kernel_size=(3, 3), input_shape=(None, 64, 64, 40),
              padding='same', return_sequences=True,kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
              bias_initializer=tf_v1.keras.regularizers.l2(weight_decay))
        self.bn_3 = tf_v1.keras.layers.BatchNormalization()
        self.cl_4 = tf_v1.keras.layers.ConvLSTM2D(filters=40, strides=1, kernel_size=(3, 3), input_shape=(None, 64, 64, 40),
              padding='same', return_sequences=True, kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
              bias_initializer=tf_v1.keras.regularizers.l2(weight_decay))
        self.bn_4 = tf_v1.keras.layers.BatchNormalization()
        self.conv = tf_v1.keras.layers.Conv3D(filters=3, strides=1, kernel_size=(3, 3), kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
              bias_initializer=tf_v1.keras.regularizers.l2(weight_decay))

    def call(self, inputs):
        # Pass inputs to layers
        x = self.cl_1(inputs)
        x = self.bn_1(x)
        x = self.cl_2(x)
        x = self.bn_2(x)
        x = self.cl_3(x)
        x = self.bn_3(x)
        x = self.cl_4(x)
        x = self.bn_4(x)
        x = self.conv(x)

        input_fn = lambda: x
        out = self._generator.predict(input_fn)
        return out

