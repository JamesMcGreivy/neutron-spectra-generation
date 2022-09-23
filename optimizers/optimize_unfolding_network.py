import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import ShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping
from bayes_opt import BayesianOptimization, Events
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import sys
sys.path.append('/g/g15/mcgreivy/testSpectraGen/')

import unfolding_network
import data_generation

def evaluate_model(lr, layer1, layer2, layer3, alpha, drop, batch_size):
    
    xdata, ydata = data_generation.x_data_IAEA, data_generation.y_data_IAEA
    
    SPLITS = 6
    
    boot = ShuffleSplit(n_splits = SPLITS, test_size = 0.2)
    
    benchmark = []
    
    for train, test in boot.split(xdata):
        
        tf.keras.backend.clear_session()

        xtrain, ytrain = xdata[train], ydata[train]
        xtest, ytest = xdata[test], ydata[test]
        
        model = unfolding_network.generate_model(layer1, layer2, layer3, alpha, drop, batch_size, lr)
        
        model.compile(loss = "mse", optimizer = Adam(learning_rate=lr))
        
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                patience=40, verbose=0, mode='min',
                                restore_best_weights=True)
        
        model.fit(xtrain,ytrain,validation_data=(xtest,ytest),
                  callbacks=[monitor],verbose=0,epochs=5000,
                  batch_size=int(batch_size))
        
        pred = model.predict(xtest)
        
        score = metrics.mean_squared_error(ytest, pred)
        benchmark.append(score)
    
    print("{} ± {}".format(np.mean(benchmark), np.std(benchmark)))
    return -np.mean(benchmark)

pbounds = {"alpha" : (0.22, 0.3),
           "drop" : (0.18, 0.26),
           "layer1" : (130, 170),
           "layer2" : (130, 170),
           "layer3" : (200, 320),
           "lr" : (0.004, 0.01),
           "batch_size" : (80, 120)}

def optimize_model(init_points = 50, n_iter = 50):    

    optimizer = BayesianOptimization(f = evaluate_model,
                                     pbounds = pbounds,
                                     verbose = 2)

    logger = JSONLogger(path = constants.LOG_PATH_BO + "unfolding_opt.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points = init_points, n_iter = n_iter)

    return optimizer.max


"""
{'target': -0.00017183281017987438, 'params': {
'alpha': 0.26446072560404577, 
'batch_size': 100.026477113451, 
'drop': 0.22558178249442118, '
layer1': 151.99770439379932, 
'layer2': 146.64174582449738, 
'layer3': 296.73741284952047, 
'lr': 0.0076023538534310394
}}
"""


