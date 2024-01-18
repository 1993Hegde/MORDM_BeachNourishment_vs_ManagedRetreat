from inputs_and_functions import *
import os
import pickle
from sklearn.neighbors import KernelDensity
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import numpy as np

number_of_sow = 100
slr_scenario = 1
pars = return_input_parameters_to_model(slr_scenario)
b_range = np.random.uniform(0.000900,0.001000,number_of_sow)
delta_range = np.random.uniform(1,7,number_of_sow)
alpha_range = np.random.uniform(1.0,1.2,number_of_sow)
beta_range = np.random.uniform(0.1,1.0,number_of_sow)
d_range = np.random.uniform(-0.05,0.05,number_of_sow)
A_range = np.random.uniform(0.7*669000,1.3*669000,number_of_sow)
W_range = np.random.uniform(0.7*500000,1.3*500000,number_of_sow)
epsilon_range = np.random.uniform(0,0.1,number_of_sow)
xi_range = np.random.uniform(0,0.1,number_of_sow)
l_range = np.random.uniform(0.176,0.264,number_of_sow)
eta_range = np.random.uniform(824,1.5*824,number_of_sow)

pars["b"]=np.quantile(b_range,0.75)
pars["delta"]=np.quantile(delta_range,0.75)
pars["alpha"] = np.quantile(alpha_range,0.25)
pars["beta"] = np.quantile(beta_range,0.75)
pars["d"] = np.median(d_range)
pars["A"]=np.quantile(A_range,0.25)
pars["W"]=np.quantile(W_range,0.25)
pars["epsilon"]=np.quantile(epsilon_range,0.75)
pars["xi"]=np.quantile(xi_range,0.75)
pars["l"]=np.quantile(l_range,0.25)
pars["eta_range"]=np.median(eta_range)
dir_name = "100_SoW_30_time_steps_wo_exponential_DamageFunction"
os.mkdir(dir_name)

sow_number =0
total_sows = number_of_sow
npv_dict = {}
os.chdir(dir_name)
while sow_number<total_sows:
    print("#######################################")
    print("I am entering SoW " + str(sow_number+1))
    os.mkdir("SoW_"+str(sow_number+1))
    os.chdir("SoW_"+str(sow_number+1))
    pars["b"] = b_range[sow_number]
    pars["delta"] = delta_range[sow_number]
    pars["alpha"] = alpha_range[sow_number]
    pars["beta"] = beta_range[sow_number]
    pars["d"] = d_range[sow_number]
    pars["A"] = A_range[sow_number]
    pars["W"] = W_range[sow_number]
    pars["epsilon"] = epsilon_range[sow_number]
    pars["xi"] = xi_range[sow_number]
    pars["l"] = l_range[sow_number]
    pars["Cfunc"] = "concave"
    pars["Cfunc"] = random.choice(["linear", "concave", "polynomial", "exponential"])
    with open('pars.pickle', 'wb') as handle:
        pickle.dump(pars, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
    print("I am exiting SoW " + str(sow_number+1))
    print("#######################################")
    sow_number+=1
    os.chdir("..") 
os.chdir("..")