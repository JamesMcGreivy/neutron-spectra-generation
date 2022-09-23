import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Lambda, Flatten, Reshape, LeakyReLU, Dense, Activation, Dropout, InputLayer, Input
from tensorflow.keras.layers import ReLU
import tensorflow.keras.initializers
import statistics
import tensorflow.keras as keras

import constants
import data_generation

def generate_model(layer1, layer2, layer3, alpha, drop, batch_size, lr):
    
    model = Sequential()
    
    model.add(Dense(int(layer1), activation = LeakyReLU(alpha = alpha)))
    model.add(Dropout(drop))
    model.add(Dense(int(layer2), activation = LeakyReLU(alpha = alpha)))
    model.add(Dropout(drop))
    model.add(Dense(int(layer3), activation = LeakyReLU(alpha = alpha)))
    model.add(Dropout(drop))
    model.add(Dense(int(data_generation.yDim), activation = "softmax"))
    
    return model
