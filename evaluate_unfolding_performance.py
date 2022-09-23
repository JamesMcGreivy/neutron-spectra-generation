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


xeval, yeval = None, None
xother, yother = None, None

def shuffle_test_data():
    global xeval, yeval, xother, yother
    x,y = data_generation.x_data_IAEA, data_generation.y_data_IAEA
    boot = ShuffleSplit(n_splits = 1, test_size = 0.5)
    for other, test in boot.split(x):
        xeval, yeval = x[test], y[test]
        xother, yother = x[other], y[other]
        return

shuffle_test_data()

def evaluate(algorithm):
    
    trials = 3
    
    benchmark = []
    models = []
    
    data_points = 500
    xdata, ydata = algorithm(data_points, yother)
    
    boot = ShuffleSplit(n_splits = trials, test_size = 0.2)
    trial = 0
    for train, test in boot.split(xdata):
        
        trial += 1
        
        xtrain, ytrain = xdata[train], ydata[train]
        xtest, ytest = xdata[test], ydata[test]
        
        print("Starting {} ...".format(trial))
        
        model = unfolding_network.generate_model(**constants.OPT_PARAMS_UNFOLD)
        
        model.compile(loss = "mse", optimizer = Adam(learning_rate=constants.OPT_PARAMS_UNFOLD["lr"]))
        
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                patience=500, verbose=0, mode='min',
                                restore_best_weights=True)
        
        model.fit(xtrain,ytrain,validation_data=(xtest,ytest),
                  callbacks=[monitor],verbose=0,epochs=5000,
                  batch_size=int(constants.OPT_PARAMS_UNFOLD["batch_size"]))
        
        pred = model.predict(xeval)
        
        score = metrics.mean_squared_error(yeval, pred)
        
        benchmark.append(score)
        
        models.append(model)
    
    return np.mean(benchmark), np.std(benchmark), models
    
    
    