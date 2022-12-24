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

# This is the general unfolding model used throughout the paper.
# - 1 input layer (implied)
# - 1 hidden layer
# - 1 dropout layer
# - 1 hidden layer
# - 1 dropout layer
# - 1 output layer (linear activation for regression)

def generate_model(layer1, layer2, alpha, drop, batch_size):
    
    model = Sequential()
    
    model.add(Dense(int(layer1), activation = LeakyReLU(alpha = alpha)))
    model.add(Dropout(drop))
    model.add(Dense(int(layer2), activation = LeakyReLU(alpha = alpha)))
    model.add(Dropout(drop))
    model.add(Dense(int(data_generation.yDim), activation = "linear"))
    
    return model
