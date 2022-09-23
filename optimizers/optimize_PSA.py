import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import ShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import sys
sys.path.append('/g/g15/mcgreivy/testSpectraGen/')

import unfolding_network
import data_generation
import constants

# PSA == Perturbed Spectra Algorithm

def eval_PSA_params(stretch_sigma, noise):
    
    TRIALS = 6

    benchmark = []

    x_data_IAEA, y_data_IAEA = data_generation.x_data_IAEA, data_generation.y_data_IAEA
    
    for __ in range(TRIALS):

        tf.keras.backend.clear_session()

        # Split the IAEA data in half -- use half at the end to evaluate performance
        # and use half to perturb from.
        xeval, yeval = None, None
        xperturb, yperturb = None, None

        boot = ShuffleSplit(n_splits = 1, test_size = 0.5)
        for eval_i, perturb_i in boot.split(x_data_IAEA):
            xeval, yeval = x_data_IAEA[eval_i], y_data_IAEA[eval_i]
            xperturb, yperturb = x_data_IAEA[perturb_i], y_data_IAEA[perturb_i]


        xdata, ydata = data_generation.perturb_spectra_algorithm(stretch_sigma, noise, yperturb)

        # Split the perturbed data into training and testing / validation data
        xtrain, ytrain = None, None
        xtest, ytest = None, None

        boot = ShuffleSplit(n_splits = 1, test_size = 0.2)
        for train_i, test_i in boot.split(xdata):
            xtrain, ytrain = xdata[train_i], ydata[train_i]
            xtest, ytest = xdata[test_i], ydata[test_i]

        model = unfolding_network.generate_model(**constants.OPT_PARAMS_UNFOLD)
        
        model.compile(loss = "mse", optimizer = Adam(learning_rate=constants.OPT_PARAMS_UNFOLD["lr"]))
        
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                patience=40, verbose=0, mode='min',
                                restore_best_weights=True)
        
        model.fit(xtrain,ytrain,validation_data=(xtest,ytest),
                  callbacks=[monitor],verbose=0,epochs=5000,
                  batch_size=int(constants.OPT_PARAMS_UNFOLD["batch_size"]))
        
        pred = model.predict(xeval)
        
        score = metrics.mean_squared_error(yeval, pred)
        benchmark.append(score)
    
    print("{} ± {}".format(np.mean(benchmark), np.std(benchmark)))
    return -np.mean(benchmark)

pbounds = {"stretch_sigma" : (0, 0.8),
           "noise" : (0, 0.8)}

def optimize_model(init_points = 50, n_iter = 50):    

    optimizer = BayesianOptimization(f = eval_PSA_params,
                                     pbounds = pbounds,
                                     verbose = 2)

    logger = JSONLogger(path = constants.LOG_PATH_BO + "PSA_opt.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points = init_points, n_iter = n_iter)

    return optimizer.max

"""
{'target': -0.00022878263066039348, 'params': 
{'noise': 0.5333230558957331, 'stretch_sigma': 0.9451787749620012}

{'target': -0.00027627082223175323, 'params': 
{'noise': 0.47698174697632445, 'stretch_sigma': 0.8407690206574538}}
{'-0.000261 |  0.2121   |  0.3055d'}
"""


