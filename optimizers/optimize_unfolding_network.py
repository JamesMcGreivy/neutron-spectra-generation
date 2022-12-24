import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import ShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping
from bayes_opt import BayesianOptimization, Events
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Must properly set to home directory of "NeutronSpectraGeneration" 
import sys
sys.path.append("/g/g15/mcgreivy/NeutronSpectraGeneration/")

import unfolding_network
import data_generation
import constants

# General purpose optimizer, for determining the hyperparameters
# of an unfolding network trained on a given dataset. This uses
# the technique of bayesian hyperparameter tuning.
# - Inputs : the hyperparameter of the unfolding network, 
#            the data generation algorithm to train the network on.
#            To simulate an experiment with no real-world validation data, the hyperparamters of the neural network
#            are only converged using the simulated data, instead of using the IAEA data.
# - Output : the average performance of these hyperparameters in unfolding
#            data which the unfolding network has not been explicitly trained on.
def evaluate_model(layer1, layer2, alpha, drop, batch_size, data_generator):
    
    lr = 0.001
    
    xdata, ydata = data_generator(1000, data_generation.y_data_IAEA)
    
    SPLITS = 5
    
    boot = ShuffleSplit(n_splits = SPLITS, test_size = 0.2)
    
    benchmark = []
    
    for train, test in boot.split(xdata):

        xtrain, ytrain = xdata[train], ydata[train]
        xtest, ytest = xdata[test], ydata[test]
        
        model = unfolding_network.generate_model(layer1, layer2, alpha, drop, batch_size)
        
        model.compile(loss = "mse", optimizer = Adam(learning_rate=lr))
        
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-10,
                                patience=5, verbose=0, mode='min',
                                restore_best_weights=True)
        
        model.fit(xtrain,ytrain,validation_data=(xtest,ytest),
                  callbacks=[monitor],verbose=0,epochs=10000,
                  batch_size=int(batch_size))
        
        pred = model.predict(xtest)
        score = metrics.mean_squared_error(ytest, pred)
        benchmark.append(score)
    
    print("{} ± {}".format(np.mean(benchmark), np.std(benchmark)))
    return -1e6 * np.mean(benchmark)

def optimize_model(data_generator_type, bound_scale = 2, init_points = 50, n_iter = 50):    
    
    # The data generation algorithm being used to train the network.
    data_generator = None
    
    # Pulls the previous best determined hyperparameter bounds from the constants file
    bounds = None

    if data_generator_type in "FRUIT":
        data_generator = data_generation.FRUIT
        bounds = constants.OPT_PARAMS_UNFOLD_FRUIT

    if data_generator_type in "RAND":
        data_generator = data_generation.RAND
        bounds = constants.OPT_PARAMS_UNFOLD_RAND

    if data_generator_type in "GAN":
        data_generator = data_generation.GAN
        bounds = constants.OPT_PARAMS_UNFOLD_GAN

    if data_generator_type in "GAUSS":
        data_generator = data_generation.GAUSS
        bounds = constants.OPT_PARAMS_UNFOLD_GAUSS
        
    if data_generator_type in "GAUSS_LINEAR":
        data_generator = data_generation.GAUSS_LINEAR
        bounds = constants.OPT_PARAMS_UNFOLD_GAUSS_LINEAR
        
    if data_generator_type in "GAUSS_SKEW":
        data_generator = data_generation.GAUSS_SKEW_LINEAR
        bounds = constants.OPT_PARAMS_UNFOLD_GAUSS_SKEW
        
    if data_generator_type in "GAUSS_SKEW_LINEAR":
        data_generator = data_generation.GAUSS_LINEAR
        bounds = constants.OPT_PARAMS_UNFOLD_GAUSS_SKEW_LINEAR
        
    if data_generator_type in "IAEA":
        data_generator = lambda num, y_perturb : data_generation.loadXY()
        bounds = constants.OPT_PARAMS_UNFOLD
        
    pbounds = {"alpha" : (bounds["alpha"] / bound_scale, bounds["alpha"] * bound_scale),
               "drop" : (bounds["drop"] /  bound_scale, bounds["drop"] * bound_scale),
               "layer1" : (bounds["layer1"] / bound_scale, bounds["layer1"] * bound_scale),
               "layer2" : (bounds["layer2"] / bound_scale, bounds["layer2"] * bound_scale),
               "layer3" : (bounds["layer3"] / bound_scale, bounds["layer3"] * bound_scale),
               "batch_size" : (bounds["batch_size"] / bound_scale, bounds["batch_size"] * bound_scale)}

    f = lambda layer1, layer2, layer3, alpha, drop, batch_size : evaluate_model(layer1, layer2, layer3, alpha, drop, batch_size, data_generator)
    optimizer = BayesianOptimization(f = f,
                                     pbounds = pbounds,
                                     verbose = 2)

    logger = JSONLogger(path = constants.LOG_PATH_BO + data_generator_type + "_unfolding_opt.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points = init_points, n_iter = n_iter)

    return optimizer.max
"""



