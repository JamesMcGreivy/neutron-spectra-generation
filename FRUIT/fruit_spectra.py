# Move into the proper directory on LASSEN
import os
os.chdir("/g/g15/mcgreivy/NeutronSpectraGeneration/")
import sys
sys.path.append('/g/g15/mcgreivy/NeutronSpectraGeneration/FRUIT')
sys.path.append('/g/g15/mcgreivy/NeutronSpectraGeneration')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import fruit_fittings
import data_generation
import constants
import random
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import data_generation
import constants

fittings = fruit_fittings.FRUIT_FITTINGS

T_0 = 2.53e-8
E_d = 7.07e-8

def thermal(E, T_0):
    spectra = ( (E / (np.power(T_0, 2)))*np.exp(-E/T_0) )
    
    if np.sum(spectra) == 0:
        return np.zeros(len(E))
    return spectra / np.sum(spectra)

def epithermal(E, E_d, b, beta_p):
    spectra = ( (1 - np.exp( -np.power((E/E_d), 2) )) * np.power(E, (b-1)) * np.exp(-E/beta_p) )
    
    if np.sum(spectra) == 0:
        return np.zeros(len(E))
    return spectra / np.sum(spectra)
    
def fast_fission(E, alpha, beta):
    spectra = (np.power(E, alpha)) * np.exp(-E / beta)
    
    if np.sum(spectra) == 0:
        return np.zeros(len(E))
    return spectra / np.sum(spectra)

def fast_evaporation(E, T_ev):
    spectra = (E / (np.power(T_ev, 2))) * np.exp(-E / T_ev)
    
    if np.sum(spectra) == 0:
        return np.zeros(len(E))
    return spectra / np.sum(spectra)

def fast_gaussian(E, E_m, sigma):
    spectra = np.exp( - np.power((E - E_m), 2) / (2 * np.power((sigma * E_m), 2) ) )
    
    if np.sum(spectra) == 0:
        return np.zeros(len(E))
    return spectra / np.sum(spectra)

def fast_high_energy(E, T_ev):
    spectra = (E / (np.power(T_ev, 2))) * np.exp(-E / T_ev)
    
    if np.sum(spectra) == 0:
        return np.zeros(len(E))
    return spectra / np.sum(spectra)

def high_energy_func(E, T_hi):
    spectra = (E / (np.power(T_hi, 2))) * np.exp(-E / T_hi)
    
    if np.sum(spectra) == 0:
        return np.zeros(len(E))
    return spectra / np.sum(spectra)


def fission(E, P_th, P_e, b, beta_p, alpha, beta):
    E = E * 1e-6
    while P_th + P_e > 1:
        P_th = P_th / 2.0
        P_e = P_e / 2.0
    P_f = 1 - P_th - P_e
    
    spectra = P_th * thermal(E, T_0) + P_e * epithermal(E, E_d, b, beta_p) + P_f * fast_fission(E, alpha, beta)
    return spectra / np.sum(spectra)

def evaporation(E, P_th, P_e, b, beta_p, T_ev):
    E = E * 1e-6
    while P_th + P_e > 1:
        P_th = P_th / 2.0
        P_e = P_e / 2.0
    P_f = 1 - P_th - P_e
    
    spectra = P_th * thermal(E, T_0) + P_e * epithermal(E, E_d, b, beta_p) + P_f * fast_evaporation(E, T_ev)
    return spectra / np.sum(spectra)

def gaussian(E, P_th, P_e, b, beta_p, E_m, sigma):
    E = E * 1e-6
    while P_th + P_e > 1:
        P_th = P_th / 2.0
        P_e = P_e / 2.0
    
    P_f = 1 - P_th - P_e
    
    spectra = P_th * thermal(E, T_0) + P_e * epithermal(E, E_d, b, beta_p) + P_f * fast_gaussian(E, E_m, sigma)
    return spectra / np.sum(spectra)

def high_energy(E, P_th, P_e, P_f, b, beta_p, T_ev, T_hi):
    E = E * 1e-6
    while P_th + P_e + P_f > 1:
        P_th = P_th / 2.0
        P_e = P_e / 2.0
        P_f = P_f / 2.0
    P_he = 1 - (P_th + P_e + P_f)
    
    if T_hi < T_ev:
        T_hi = 10**(np.log10(T_ev / T_hi))
    
    spectra = P_th * thermal(E, T_0) + P_e * epithermal(E, E_d, b, beta_p) + P_f * fast_high_energy(E, T_ev) + P_he * high_energy_func(E, T_hi)
    return spectra / np.sum(spectra)


P_th_fission, P_e_fission, b_fission, beta_p_fission, alpha_fission, beta_fission = [], [], [], [], [], [] 
def randFission():
	# If never called
	if len(P_th_fission) == 0:
		for fit in fittings:
		    if fit["fit_type"] in "fission":
		        while (fit["params"][0] + fit["params"][1]) > 1.0:
		            fit["params"][0] = fit["params"][1] / 2.0
		            fit["params"][1] = fit["params"][1] / 2.0
		        P_th_fission.append(fit["params"][0]) 
		        P_e_fission.append(fit["params"][1]) 
		        b_fission.append(fit["params"][2]) 
		        beta_p_fission.append(fit["params"][3]) 
		        alpha_fission.append(fit["params"][4]) 
		        beta_fission.append(fit["params"][5])
	
	P_th_rand = random.choice(P_th_fission)
	P_e_rand = P_e_fission[P_th_fission.index(P_th_rand)]
	b_rand = random.choice(b_fission)
	beta_p_rand = random.choice(beta_p_fission)
	alpha_rand = random.choice(alpha_fission)
	beta_rand = random.choice(beta_fission)
    
    
	# fission(E, P_th, P_e, T_0, E_d, b, beta_p, alpha, beta):
	return fission(constants.Ebins, P_th_rand, P_e_rand, b_rand, beta_p_rand, alpha_rand, beta_rand)


P_th_gauss, P_e_gauss, b_gauss, beta_p_gauss, E_m_gauss, sigma_gauss = [], [], [], [], [], []
def randGauss():
	# If never called
	if len(P_th_gauss) == 0:
		for fit in fittings:
		    if fit["fit_type"] in "gaussian":
		        if (fit["params"][0] + fit["params"][1]) > 1.0:
		            fit["params"][0] = fit["params"][0] / 2.0
		            fit["params"][1] = fit["params"][1] / 2.0
		        P_th_gauss.append(fit["params"][0]) 
		        P_e_gauss.append(fit["params"][1]) 
		        b_gauss.append(fit["params"][2]) 
		        beta_p_gauss.append(fit["params"][3]) 
		        E_m_gauss.append(fit["params"][4])
		        sigma_gauss.append(fit["params"][5])
	P_th_rand = random.choice(P_th_gauss)
	P_e_rand = P_e_gauss[P_th_gauss.index(P_th_rand)]
	b_rand = random.choice(b_gauss)
	beta_p_rand = random.choice(beta_p_gauss)
	E_m_rand = random.choice(E_m_gauss)
	sigma_rand = random.choice(sigma_gauss)
    
	# gaussian(E, P_th, P_e, T_0, E_d, b, beta_p, E_m, sigma):
	return gaussian(constants.Ebins, P_th_rand, P_e_rand, b_rand, beta_p_rand, E_m_rand, sigma_rand)


P_th_high_energy, P_e_high_energy, P_f_high_energy, b_high_energy, beta_p_high_energy, T_ev_high_energy, T_hi_high_energy = [], [], [], [], [], [], []
def randHighEnergy():
	# If never called
	if len(P_th_high_energy) == 0:
		for fit in fittings:
		    if fit["fit_type"] in "high_energy":
		        while (fit["params"][0] + fit["params"][1] + fit["params"][2]) > 1.0:
		            fit["params"][0] = fit["params"][0] / 2.0
		            fit["params"][1] = fit["params"][1] / 2.0
		            fit["params"][2] = fit["params"][2] / 2.0    
		        P_th_high_energy.append(fit["params"][0]) 
		        P_e_high_energy.append(fit["params"][1])
		        P_f_high_energy.append(fit["params"][2])
		        b_high_energy.append(fit["params"][3]) 
		        beta_p_high_energy.append(fit["params"][4]) 
		        T_ev_high_energy.append(fit["params"][5])
		        T_hi_high_energy.append(fit["params"][6])
		    if fit["fit_type"] in "evaporation":
		        while (fit["params"][0] + fit["params"][1]) > 1.0:
		            fit["params"][0] = fit["params"][0] / 2.0
		            fit["params"][1] = fit["params"][1] / 2.0
		        P_th_high_energy.append(fit["params"][0]) 
		        P_e_high_energy.append(fit["params"][1])
		        P_f_high_energy.append(1.0 - (fit["params"][0] + fit["params"][1]))
		        b_high_energy.append(fit["params"][2]) 
		        beta_p_high_energy.append(fit["params"][3]) 
		        T_ev_high_energy.append(fit["params"][4])
    
	P_th_rand = random.choice(P_th_high_energy)
	P_e_rand = P_e_high_energy[P_th_high_energy.index(P_th_rand)]
	P_f_rand = P_f_high_energy[P_th_high_energy.index(P_th_rand)]
	b_rand = random.choice(b_high_energy)
	beta_p_rand = random.choice(beta_p_high_energy)
	T_ev_rand = random.choice(T_ev_high_energy)
	T_hi_rand = random.choice(T_hi_high_energy)

    #high_energy(E, P_th, P_e, P_f, T_0, E_d, b, beta_p, T_ev, T_hi):
	return high_energy(constants.Ebins, P_th_rand, P_e_rand, P_f_rand, b_rand, beta_p_rand, T_ev_rand, T_hi_rand)
    
def randEvap():
	# If never called
	if len(P_th_high_energy) == 0:
		for fit in fittings:
		    if fit["fit_type"] in "high_energy":
		        while (fit["params"][0] + fit["params"][1] + fit["params"][2]) > 1.0:
		            fit["params"][0] = fit["params"][0] / 2.0
		            fit["params"][1] = fit["params"][1] / 2.0
		            fit["params"][2] = fit["params"][2] / 2.0    
		        P_th_high_energy.append(fit["params"][0]) 
		        P_e_high_energy.append(fit["params"][1])
		        P_f_high_energy.append(fit["params"][2])
		        b_high_energy.append(fit["params"][3]) 
		        beta_p_high_energy.append(fit["params"][4]) 
		        T_ev_high_energy.append(fit["params"][5])
		        T_hi_high_energy.append(fit["params"][6])
		    if fit["fit_type"] in "evaporation":
		        while (fit["params"][0] + fit["params"][1]) > 1.0:
		            fit["params"][0] = fit["params"][0] / 2.0
		            fit["params"][1] = fit["params"][1] / 2.0
		        P_th_high_energy.append(fit["params"][0]) 
		        P_e_high_energy.append(fit["params"][1])
		        P_f_high_energy.append(1.0 - (fit["params"][0] + fit["params"][1]))
		        b_high_energy.append(fit["params"][2]) 
		        beta_p_high_energy.append(fit["params"][3]) 
		        T_ev_high_energy.append(fit["params"][4])
		        
    
	P_th_rand = random.choice(P_th_high_energy)
	P_e_rand = P_e_high_energy[P_th_high_energy.index(P_th_rand)]
	b_rand = random.choice(b_high_energy)
	beta_p_rand = random.choice(beta_p_high_energy)
	T_ev_rand = random.choice(T_ev_high_energy)
    
	# evaporation(E, P_th, P_e, T_0, E_d, b, beta_p, T_ev):
	return evaporation(constants.Ebins, P_th_rand, P_e_rand, b_rand, beta_p_rand, T_ev_rand)


