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

def evaluate_GPA_params(meanPeakCenter, stdPeakCenter, meanPeakWidth, 
                      stdPeakWidth, extraPeakProb, noise, ampDecay):
    
    SPLITS = 6
    benchmark = []

    xeval, yeval = data_generation.x_data_IAEA, data_generation.y_data_IAEA

    xdata, ydata = data_generation.gaussian_peak_algorithm(meanPeakCenter, stdPeakCenter, meanPeakWidth, 
                                                                stdPeakWidth, extraPeakProb, noise, ampDecay)
    
    boot = ShuffleSplit(n_splits = SPLITS, test_size = 0.2)
    for train, test in boot.split(xdata):

        tf.keras.backend.clear_session()

        xtrain, ytrain = xdata[train], ydata[train]
        xtest, ytest = xdata[test], ydata[test]
        
        model = unfolding_network.generate_model(**constants.OPT_PARAMS_UNFOLD)
        
        model.compile(loss = "mse", optimizer = Adam(learning_rate=constants.OPT_PARAMS_UNFOLD["lr"]))
        
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5,
                                patience=40, verbose=0, mode='min',
                                restore_best_weights=True)
        
        model.fit(xtrain,ytrain,validation_data=(xtest,ytest),
                  callbacks=[monitor],verbose=0,epochs=5000,
                  batch_size=int(constants.OPT_PARAMS_UNFOLD["batch_size"]))
        
        pred = model.predict(xeval)
        
        score = metrics.mean_squared_error(yeval, pred)
        benchmark.append(score)
        
    print("{} ± {}".format(np.mean(benchmark), 2 * np.std(benchmark)))
        
    return -np.mean(benchmark)

pbounds = {"ampDecay" : (0.1, 0.9),
           "extraPeakProb" : (0.1, 0.9),
           "meanPeakCenter" : (-7, -15),
           "meanPeakWidth" : (0, 10),
           "noise" : (0, 0),
           "stdPeakCenter" : (0, 10),
           "stdPeakWidth" : (0, 5)}

def optimize_model(file_name, init_points = 50, n_iter = 50):    

    optimizer = BayesianOptimization(f = evaluate_GPA_params,
                                    pbounds = pbounds,
                                    verbose = 2)

    logger = JSONLogger(path = constants.LOG_PATH_BO + file_name + ".json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points = init_points, n_iter = n_iter)

    return optimizer.max

"""
 target   | ampDecay  | extraP... | meanPe... | meanPe... |   noise   | stdPea... | stdPea... |
-0.00049  |  0.1      |  0.9      | -0.3927   |  2.911    |  0.0      |  5.428    |  4.0      |

{'target': -0.00048305683938511143, 'params': {
'ampDecay': 0.15403714272221045, 
'extraPeakProb': 0.7964612516930676, 
'meanPeakCenter': -1.1916863728628457, 
'meanPeakWidth': 2.219829939422177, 
'noise': 0.25677306014586154, 
'stdPeakCenter': 6.36610572587569, 
'stdPeakWidth': 5.839641275287087}
}
"""


