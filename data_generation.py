import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import constants
import os
from tensorflow.keras.models import model_from_json
import scipy.interpolate
import scipy
import random

import GAN_network

import sys
sys.path.append('/g/g15/mcgreivy/NeutronSpectraGeneration/FRUIT')
sys.path.append('/g/g15/mcgreivy/NeutronSpectraGeneration')

import fruit_spectra

# Loads the real-world data from IAEA tecdoc
def loadXY():

    '''
    Can't figure out how to pass non-optimiziable variable to the bayes opt
    module. Just load them from pickle everytime in evaluate_network()
    '''
    unfolding_data = pd.read_pickle(constants.HOME_DIR + "IAEA_data/unfolding_data.pkl")

    X = np.zeros((251,15))
    Y = np.zeros((251,53))
    for row in range(251):
        #X[row,:] = unfolding_data['Detector Response'][row]
        Y[row,:] = unfolding_data['Spectrum'][row][0:53]
    
    # Divides out the lethargy interval
    for y in Y:
        for i in range(len(constants.Ebins)):
            y[i] = y[i] * constants.lethargies[i]
    
    # Fixes normalization back to 1 because of precision errors
    for i in range(len(Y)):
        Y[i] = Y[i] / np.sum(Y[i])
    
    # Multiplies by the conversion matrix to determine the detector response
    X = []
    for y in Y:
        X.append(np.matmul(constants.cm, y))
    X = np.array(X)
    
    return X,Y

x_data_IAEA,y_data_IAEA = loadXY()
yDim = y_data_IAEA.shape[-1]
NUM_DATA = len(y_data_IAEA)

# Perturbed Spectra Algorithm
# stretch sigma -> standard deviation in the stretching factor applied for a perturbation
# noise -> maximum noise amplitude to add on top of the perturbed spectra
# yPerturbFrom -> the data set of spectra to perturb from

def perturb_spectra_algorithm(stretch_sigma, noise, yPerturbFrom, NUM_DATA = NUM_DATA):
    
    def perturb_spectra(spectra_original):
        
        spectra = spectra_original
        # First, stretch the spectra by some amount
        f = scipy.interpolate.interp1d(constants.Ebins, spectra, bounds_error = False, fill_value = (0,0))
        spectra = f(constants.Ebins * np.abs(np.random.normal(loc = 1, scale = stretch_sigma)))

        # Then, add some noise
        spectra = spectra * (1 + (noise * np.random.random(len(spectra))))

        # Then, re-normalize
        if np.sum(spectra) == 0:
            return perturb_spectra(spectra_original)
        
        spectra = spectra / np.sum(spectra)
        return spectra
    
    indexes = np.floor(len(yPerturbFrom) * np.random.random(NUM_DATA))
    indexes = [int(i) for i in indexes]
    spectras = []
    for i in indexes:
        spectras.append(yPerturbFrom[i])
    spectras = np.array(spectras)
    
    xData = []
    yData = []
    for spectra in spectras:
        y = perturb_spectra(spectra)
        yData.append(y)
        xData.append(np.matmul(constants.cm, y))
        
    xData = np.array(xData)
    yData = np.array(yData)
    
    return xData, yData

PSA = lambda num_data, yPerturbFrom : perturb_spectra_algorithm(constants.OPT_PARAMS_PSA["stretch_sigma"], constants.OPT_PARAMS_PSA["stretch_sigma"], yPerturbFrom, num_data)


# FRUIT Algorithm
def FRUIT_spectra_algorithm(NUM_DATA = NUM_DATA):

    yData = []
    for i in range(NUM_DATA):
        spectra_type = random.randint(0, 3)

        if spectra_type == 0:
            yData.append(fruit_spectra.randFission())
        elif spectra_type == 1:
            yData.append(fruit_spectra.randEvap())
        elif spectra_type == 2:
            yData.append(fruit_spectra.randGauss())
        else:
            yData.append(fruit_spectra.randHighEnergy())

    yData = np.array(yData)
    xData = []
    for y in yData:
        xData.append(np.matmul(constants.cm, y))

    return np.array(xData), yData

FRUIT = lambda num_data, yPerturbFrom : FRUIT_spectra_algorithm(num_data)


# GAN Algorithm
def setup_GAN():
    GAN_network.construct_GAN(**constants.OPT_PARAMS_GAN)
    GAN_network.train(constants.OPT_PARAMS_GAN["epochs"], constants.OPT_PARAMS_GAN["batch_size"])

def GAN_spectra_algorithm(NUM_DATA = NUM_DATA):

    if GAN_network.generator == None:
        GAN_network.load_GAN()
    
    noise = np.random.normal(0, 1, (NUM_DATA, GAN_network.SEED_SIZE))
    
    yData = np.array(GAN_network.generator(noise))

    for i in range(len(yData)):
        yData[i] = yData[i] / np.sum(yData[i])
    
    xData = []
    for y in yData:
        xData.append(np.matmul(constants.cm, y))

    return np.array(xData), yData

GAN = lambda num_data, yPerturbFrom : GAN_spectra_algorithm(num_data)

# Random Spectra

def random_spectra_algorithm(numData = NUM_DATA):
    
    yData = np.random.random(size=(numData,yDim))
    for i in range(numData):
        yData[i] = yData[i] / sum(yData[i])
        
    xData = []
    for y in yData:
        xData.append(np.matmul(constants.cm, y))
    xData = np.array(xData)
    
    return xData, yData

RAND = lambda num_data, yPerturbFrom : random_spectra_algorithm(num_data)

# Gaussian Peaks

def gaussian_peak_algorithm(meanPeakCenter, stdPeakCenter, meanPeakWidth, 
                          stdPeakWidth, extraPeakProb, noise, ampDecay,
                          numData = NUM_DATA):

    def gaussian(x, A, mean, std):
        return A * np.exp( -0.5 * ( (x - mean)**2 / std**2 ) )

    def addGaussian(x, y, A, mean, std):
        if len(x) != len(y):
            raise ValueError("Arrays Must be Equal Length")
        for i in range(len(x)):
            y[i] = y[i] + gaussian(np.log(x[i]), A, mean, std)
        return

    xData = []
    yData = []
    for i in range(numData):
        y = np.zeros(len(constants.Ebins))
        
        peakAmp = 1
        addPeak = 0
        while addPeak < extraPeakProb:
            peakCenter = np.abs(np.random.normal(loc = meanPeakCenter, scale = stdPeakCenter))
            peakWidth = np.abs(np.random.normal(loc = meanPeakWidth, scale = stdPeakWidth))
            addGaussian(constants.Ebins, y, peakAmp, peakCenter, peakWidth)
            
            addPeak = np.random.random()
            peakAmp = peakAmp * (2.0 * np.random.random()) * ampDecay 
        
        y = y * (1 + noise * np.random.random(len(y)) )
        
        if np.sum(y) == 0:
            continue
        y = y / np.sum(y)
        
        yData.append(y)
        xData.append(np.matmul(constants.cm, y))
        
    return np.array(xData), np.array(yData)
        
GAUSS = lambda num_data, yPerturbFrom : gaussian_peak_algorithm(**constants.OPT_PARAMS_GPA, numData = num_data)

