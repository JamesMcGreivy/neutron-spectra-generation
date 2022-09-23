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
import GAN_network

def eval_GAN_params(layer1, layer2, layer3, alpha, drop, lr, batch_size, epochs):
    
    if epochs > 0:
        GAN_network.construct_GAN(layer1, layer2, layer3, alpha, drop, lr, batch_size, epochs)
        GAN_network.train(epochs = int(epochs), batch_size = int(batch_size))
    
    trials = 6
    benchmark = []

    xeval, yeval = data_generation.x_data_IAEA, data_generation.y_data_IAEA

    xdata, ydata = data_generation.GAN_spectra_algorithm()

    boot = ShuffleSplit(n_splits = trials, test_size = 0.2)
    for train, test in boot.split(xdata):

        tf.keras.backend.clear_session()

        GAN_network.load_GAN()
        
        xtrain, ytrain = xdata[train], ydata[train]
        xtest, ytest = xdata[test], ydata[test]
        
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
    return - np.mean(benchmark)

pbounds = {"alpha" : (0.2, 0.32),
           "drop" : (0.12, 0.24),
           "layer1" : (110, 130),
           "layer2" : (110, 130),
           "layer3" : (330, 350),
           "lr" : (0.0005, 0.005),
           "batch_size" : (150, 190),
           "epochs" : (300, 350)}

def optimize_model(init_points = 50, n_iter = 50):    

    optimizer = BayesianOptimization(f = eval_GAN_params,
                                     pbounds = pbounds,
                                     verbose = 2)

    logger = JSONLogger(path = constants.LOG_PATH_BO + "GAN_opt.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points = init_points, n_iter = n_iter)

    return optimizer.max

"""
 target   |   alpha   | batch_... |   drop    |  epochs   |  layer1   |  layer2   |  layer3   |    lr     |
-0.000561 |  0.2653   |  168.2    |  0.1811   |  325.1    |  120.1    |  119.8    |  339.3    |  0.001365 |

"""

        