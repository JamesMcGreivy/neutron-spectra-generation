import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ReLU, Lambda, Flatten, Reshape, LeakyReLU, Dense, Activation, Dropout, InputLayer, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
import tensorflow.keras.initializers
import statistics
import tensorflow.keras as keras
import numpy as np
import random

import constants
import data_generation

# Size vector to pull spectra seeds from
SEED_SIZE = 20

generator, discriminator, combined = None, None, None

def build_generator(layer1, layer2, layer3, alpha, dropout):
    
    model = Sequential()
    
    model.add(Dense(layer1,  
                    input_shape=(SEED_SIZE,)))
    model.add(LeakyReLU(alpha = alpha))
    model.add(Dropout(dropout))
    model.add(Dense(layer2))
    model.add(LeakyReLU(alpha = alpha))
    model.add(Dropout(dropout))
    model.add(Dense(layer3))
    model.add(LeakyReLU(alpha = alpha))
    model.add(Dropout(dropout))
    
    model.add(Dense(data_generation.yDim, activation='softmax'))
    
    noise = Input(shape=(SEED_SIZE,))
    generated_spectra = model(noise)
    
    return Model(noise,generated_spectra)


def build_discriminator(layer1, layer2, layer3, alpha, dropout):
    
    model = Sequential()
    model.add(Dense(layer1,
                    input_shape = (data_generation.yDim,)))
    model.add(LeakyReLU(alpha = alpha))
    model.add(Dropout(dropout))
    model.add(Dense(layer2))
    model.add(LeakyReLU(alpha = alpha))
    model.add(Dropout(dropout))
    model.add(Dense(layer3))
    model.add(LeakyReLU(alpha = alpha))
    model.add(Dropout(dropout))
    
    model.add(Dense(1, activation='sigmoid'))
    
    input_spectra = Input(shape = (data_generation.yDim,))
    validity = model(input_spectra)
    
    return Model(input_spectra, validity)

def construct_GAN(layer1, layer2, layer3, alpha, dropout, lr, batch_size, epochs):
    global generator, discriminator, combined
    
    optimizer = Adam(lr, 0.5)
    
    discriminator = build_discriminator(layer1, layer2, layer3, alpha, dropout)
    discriminator.compile(loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    
    generator = build_generator(layer1, layer2, layer3, alpha, dropout)
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    z = Input(shape=(SEED_SIZE,))
    spectra = generator(z)
    
    discriminator.trainable = False  
    
    valid = discriminator(spectra)
    
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

def train(epochs, batch_size = 32, draw_data = lambda : (data_generation.x_data_IAEA, data_generation.y_data_IAEA)):
    epochs = int(epochs)
    batch_size = int(batch_size)

    half_batch = int(batch_size / 2)
    for epoch in range(epochs):
        
        xtrain, ytrain = draw_data()
        
        indexes = random.sample(range(len(ytrain)), half_batch)
        train_on = ytrain[indexes]
        
        noise = np.random.normal(0, 1, (half_batch, SEED_SIZE))
        
        gen_spectra = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(train_on, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_spectra, np.zeros((half_batch, 1)))
        
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, SEED_SIZE))
        
        valid_y = np.array([1] * batch_size)
        
        g_loss = combined.train_on_batch(noise, valid_y)
        
        #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    save_GAN()

save_location = constants.HOME_DIR + "saved_networks/GAN/"

def save_GAN():
    generator_json = generator.to_json()
    with open(save_location + "generator.json", "w") as json_file:
        json_file.write(generator_json)
    # serialize weights to HDF5
    generator.save_weights(save_location + "generator.h5")

    discriminator_json = discriminator.to_json()
    with open(save_location + "discriminator.json", "w") as json_file:
        json_file.write(discriminator_json)
    # serialize weights to HDF5
    discriminator.save_weights(save_location + "discriminator.h5")

    combined_json = combined.to_json()
    with open(save_location + "combined.json", "w") as json_file:
        json_file.write(combined_json)
    # serialize weights to HDF5
    combined.save_weights(save_location + "combined.h5")


def load_GAN():
    global generator, discriminator, combined

    json_file = open(save_location + 'generator.json', 'r')
    loaded_json = json_file.read()
    json_file.close()
    generator = model_from_json(loaded_json)
    # load weights into new model
    generator.load_weights(save_location + "generator.h5")

    json_file = open(save_location + 'discriminator.json', 'r')
    loaded_json = json_file.read()
    json_file.close()
    discriminator = model_from_json(loaded_json)
    # load weights into new model
    discriminator.load_weights(save_location + "discriminator.h5")

    json_file = open(save_location + 'combined.json', 'r')
    loaded_json = json_file.read()
    json_file.close()
    combined = model_from_json(loaded_json)
    # load weights into new model
    combined.load_weights(save_location + "combined.h5")


