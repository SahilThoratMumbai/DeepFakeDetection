import tensorflow as tf
import sys
import os
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
import numpy as np
from models.cnn_model import CentralDifferenceConv2D

from models.vit_model import PatchExtractor
from models.ensemble import explain_decision

@st.cache_resource
def load_models():
    custom_objects = {
        'CentralDifferenceConv2D': CentralDifferenceConv2D,
        'PatchExtractor': PatchExtractor
    }
    
    cnn_model = tf.keras.models.load_model(
        'saved_models/cnn_model.keras',
        custom_objects=custom_objects
    )
    
    vit_model = tf.keras.models.load_model(
        'saved_models/vit_model.keras',
        custom_objects=custom_objects
    )
    
    return cnn_model, vit_model

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for model input"""
    img = image.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def ensemble_predict(cnn_model, vit_model, image_array):
    """Make ensemble prediction using models"""
    cnn_pred = cnn_model.predict(image_array, verbose=0)[0]
    vit_pred = vit_model.predict(image_array, verbose=0)[0]
    return explain_decision(cnn_pred, vit_pred)