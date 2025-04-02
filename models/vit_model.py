import tensorflow as tf
from tensorflow.keras import layers, utils
import numpy as np
@utils.register_keras_serializable()
class PatchExtractor(layers.Layer):
    def __init__(self, num_patches, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        
    def build(self, input_shape):
        self.patch_size = input_shape[1] // int(self.num_patches**0.5)
        self.projection = layers.Dense(128)
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=128
        )
        super().build(input_shape)
        
    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1,1,1,1],
            padding='VALID'
        )
        patches = tf.reshape(patches, (tf.shape(patches)[0], -1, self.patch_size**2 * 3))
        patches = self.projection(patches)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return patches + self.position_embedding(positions)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches
        })
        return config