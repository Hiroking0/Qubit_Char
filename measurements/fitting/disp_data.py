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

if __name__ == "__main__":
    
    f = open('../general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    
    
    fn = askopenfilename()
    print(fn)
    print(fn[:-13] + "_json.json")
    
    #try to open file as numpy array, this means it was many patterns.
    #If fails, open it as bin file
    
    arr = np.load(fn)
    print(np.shape(arr))
    #ratio = dp.get_population_ratio(arr, params['v_threshold'])
    #print(ratio)
    dp.plot_histogram(arr)
    
    
    x = np.linspace(0, 400, num = len(arr))
    dp.plot_population_v_pattern(arr, params['v_threshold'], x)
    kb = 1.38e-23
    hbar = 1.054e-34
    wq = params['set_wq']*10**9
    T = (-hbar * 2 * np.pi * wq)/(kb*(1.5))
    print(T)
    
    
    #dp.plot_np_file(len(arr), 1, len(arr[0]), 1, fn)

