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
import pandas

def disp_sequence():
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
        plt.plot(pop)
        plt.show()



    kb = 1.38e-23
    hbar = 1.054e-34
    wq = params['set_wq']*10**9
    T = (-hbar * 2 * np.pi * wq)/(kb*(1.5))
    print(T)
    

def disp_single_sweep():
    
    file = askopenfilename()
    csvFile = pandas.read_csv(file, sep = ',', engine = 'python')
    num_patterns = max(csvFile['pattern_num']) + 1
    sweep_num = int(len(csvFile['pattern_num'])/num_patterns)
    shape = (num_patterns, sweep_num)
    
    
    sweep_param = csvFile.columns[-1]
    min_sweep = min(csvFile[sweep_param].to_list())
    max_sweep = max(csvFile[sweep_param].to_list())
    print(min_sweep, max_sweep)
    x = range(num_patterns)
    y = np.linspace(min_sweep, max_sweep, num=sweep_num)
    
    plt.subplot(2,3,1)
    chA_nosub = csvFile['chA_nosub'].to_list()
    chA_nosub = np.reshape(chA_nosub, shape)
    chA_nosub = np.transpose(chA_nosub)
    plt.pcolormesh(x, y, chA_nosub, shading = 'auto')
    plt.title("chA_nosub")
    plt.xlabel("pattern #")
    plt.ylabel(sweep_param)
    
    plt.subplot(2,3,2)
    chB_nosub = csvFile['chB_nosub'].to_list()
    chB_nosub = np.reshape(chB_nosub, shape)
    chB_nosub = np.transpose(chB_nosub)
    plt.pcolormesh(x, y, chB_nosub, shading = 'auto')
    plt.title("chB_nosub")
    plt.xlabel("pattern #")
    plt.ylabel(sweep_param)
    
    plt.subplot(2,3,3)
    mags_nosub = csvFile['mag_nosub'].to_list()
    mags_nosub = np.reshape(mags_nosub, shape)
    mags_nosub = np.transpose(mags_nosub)
    plt.pcolormesh(x, y, mags_nosub, shading = 'auto')
    plt.title("mags_nosub")
    plt.xlabel("pattern #")
    plt.ylabel(sweep_param)

    plt.subplot(2,3,4)
    chA_sub = csvFile['chA_sub'].to_list()
    chA_sub = np.reshape(chA_sub, shape)
    chA_sub = np.transpose(chA_sub)
    plt.pcolormesh(x, y, chA_sub, shading = 'auto')
    plt.title("chA_sub")
    plt.xlabel("pattern #")
    plt.ylabel(sweep_param)

    plt.subplot(2,3,5)
    chB_sub = csvFile['chB_sub'].to_list()
    chB_sub = np.reshape(chB_sub, shape)
    chB_sub = np.transpose(chB_sub)
    plt.pcolormesh(x, y, chB_sub, shading = 'auto')
    plt.title("chB_sub")
    plt.xlabel("pattern #")
    plt.ylabel(sweep_param)
    
    plt.subplot(2,3,6)
    mags_sub = csvFile['mag_sub'].to_list()
    mags_sub = np.reshape(mags_sub, shape)
    mags_sub = np.transpose(mags_sub)
    plt.pcolormesh(x, y, mags_sub, shading = 'auto')
    plt.title("mags_sub")
    plt.xlabel("pattern #")
    plt.ylabel(sweep_param)
    
    
    plt.show()


def disp_double_sweep():
    pass



if __name__ == "__main__":
    #disp_sequence()
    disp_single_sweep()