import tensorflow as tf
from tensorflow.keras import layers, utils
import numpy as np
@utils.register_keras_serializable()
class CentralDifferenceConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1,1), padding="same", activation=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            self.filters, self.kernel_size, 
            strides=self.strides, padding=self.padding
        )
        self.bn = layers.BatchNormalization()
        super().build(input_shape)
        
    def call(self, inputs):
        diff_x = inputs - tf.roll(inputs, shift=[0,1], axis=[1,2])
        diff_y = inputs - tf.roll(inputs, shift=[1,0], axis=[1,2])
        central_diff = diff_x + diff_y
        x = self.conv(central_diff)
        x = self.bn(x)
        return self.activation(x) if self.activation else x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config