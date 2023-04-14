# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:44:38 2023

@author: lqc
"""
from tkinter.filedialog import askopenfilename
import numpy as np
import yaml
import sys
sys.path.append("../../")
from lib import data_process as dp
import matplotlib.pyplot as plt
import os
import json

if __name__ == "__main__":
    
    #f = open('../general_config.yaml','r')
    #params = yaml.safe_load(f)
    #f.close()
    
    
    
    fn = askopenfilename()

    
    nf = '\\'.join(fn.split('/')[0:-1]) + "/"
    print("nf", nf)

    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f:
                with open(nf + f) as file:
                    params = json.load(file)

    #print(fn[:-13] + "_json.json")
    
    #try to open file as numpy array, this means it was many patterns.
    #If fails, open it as bin file
    
    arr = np.load(fn)
    #print(np.shape(arr))
    #ratio = dp.get_population_ratio(arr, params['v_threshold'])
    #print(ratio)
    dp.plot_histogram(arr)
    x = []
    if "rabi" in nf.lower():
        x = np.linspace(params['rabi_pulse_initial_duration'], params['rabi_pulse_end_duration'], num = len(arr))
    if "ramsey" in nf.lower():
        x = np.linspace(params['ramsey_gap_1_init'], params['ramsey_gap_1_final'], num = len(arr))
    if "t1" in nf.lower():
        x = np.linspace(params['T1_init_gap'], params['T1_final_gap'], num = len(arr))
    if "echo" in nf.lower():
        x = np.linspace(params['echo_initial_t'], params['echo_final_t'], num = len(arr))
    

    if len(x) > 1:
        pop = dp.get_population_v_pattern(arr, params['v_threshold'])




    kb = 1.38e-23
    hbar = 1.054e-34
    wq = params['set_wq']*10**9
    T = (-hbar * 2 * np.pi * wq)/(kb*(1.5))
    print(T)

