import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v1 as tf_v2
import summarizer.graph

global summary_graph
summary_graph = summarizer.graph.summary_graph

class SummaryNetwork(tf_v1.keras.Model):
    def __init__(self, weight_decay):
        super(SummaryNetwork, self).__init__()
        # Define network layers
        # with summary_graph.as_default() as graph:
        self.cl_1 = tf_v1.keras.layers.ConvLSTM2D(filters=40, strides=1, kernel_size=(3, 3),
              padding='same', return_sequences=True, kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
              bias_regularizer=tf_v1.keras.regularizers.l2(weight_decay), kernel_initializer='random_uniform',
                bias_initializer='zeros')
        self.bn_1 = tf_v1.keras.layers.BatchNormalization()
        self.cl_2 = tf_v1.keras.layers.ConvLSTM2D(filters=40, strides=1, kernel_size=(3, 3),
              padding='same', return_sequences=True, kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
              bias_regularizer=tf_v1.keras.regularizers.l2(weight_decay),kernel_initializer='random_uniform',
                bias_initializer='zeros')
        self.bn_2 = tf_v1.keras.layers.BatchNormalization()
        self.cl_3 = tf_v1.keras.layers.ConvLSTM2D(filters=40, strides=1, kernel_size=(3, 3),
              padding='same', return_sequences=True,kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
              bias_regularizer=tf_v1.keras.regularizers.l2(weight_decay), kernel_initializer='random_uniform',
                bias_initializer='zeros')
        self.bn_3 = tf_v1.keras.layers.BatchNormalization()
        self.cl_4 = tf_v1.keras.layers.ConvLSTM2D(filters=40, strides=1, kernel_size=(3, 3),
              padding='same', return_sequences=True, kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
              bias_regularizer=tf_v1.keras.regularizers.l2(weight_decay),kernel_initializer='random_uniform',
                bias_initializer='zeros')
        self.bn_4 = tf_v1.keras.layers.BatchNormalization()
        self.conv = tf_v1.keras.layers.Conv3D(filters=3, strides=1, kernel_size=(20, 3, 3), kernel_regularizer=tf_v1.keras.regularizers.l2(weight_decay),
            padding='valid', bias_regularizer=tf_v1.keras.regularizers.l2(weight_decay), kernel_initializer='random_uniform',
            bias_initializer='zeros')
        self.pad = tf_v1.keras.layers.ZeroPadding2D(padding=(1, 1))

    def call(self, inputs):

        x = self.cl_1(inputs)
        x = self.bn_1(x)
        x = self.cl_2(x)
        x = self.bn_2(x)
        x = self.cl_3(x)
        x = self.bn_3(x)
        x = self.cl_4(x)
        x = self.bn_4(x)
        x = self.conv(x)
        x = tf_v1.reshape(x, [-1, 62, 62, 3])
        x = self.pad(x)
        print(x.shape)
        return x

