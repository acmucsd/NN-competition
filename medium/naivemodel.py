import tensorflow as tf
from tensorflow.keras import models, layers, losses

class SimpleModel(tf.keras.Model):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.l1 = layers.Dense(64, activation='sigmoid', input_shape=(3,))
        self.l2 = layers.Dense(128, activation='tanh')
        self.l3 = layers.Dense(1, activation=None)

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        return x
