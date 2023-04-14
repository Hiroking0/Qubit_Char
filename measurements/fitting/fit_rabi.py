# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:29:18 2021

@author: Crow108
"""

import numpy as np
#import daq_programs
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tkinter.filedialog import askopenfilename
import yaml
from pathlib import Path

def objective_rabi(x, a, b, c, d):
	return a + (b*np.sin(2*np.pi*c*x+d))

def fit_rabi(y, init_a, init_b, init_c, init_d, num_points, max_length):
    x_data = np.linspace(0, max_length, num_points)
    y_data = y
    initial  = [init_a, init_b, init_c, init_d]
    popt, _ = curve_fit(objective_rabi, x_data, y_data, p0 = initial)
    a, b , c, d= popt
    
    new_data = objective_rabi(x_data,a,b,c,d)
    #plt.plot(x_data,objective_rabi(x_data,a,b,c,d))
    #plt.plot(x_data,y_data)
    #plt.show()
    return new_data, a, b, c, d

if __name__ == "__main__":
    f = open('../general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    
    fn = askopenfilename()
    arr = np.load(fn)
    #first get pattern avgs
    avgs = np.zeros(len(arr))
    for i in range(len(arr)):
        avgs[i] = np.average(arr[i])
        
    a = 226.6
    b = .76
    c = 1/350
    d = np.pi/2
    num_patterns = len(arr)
    longest_rabi = params['rabi_pulse_end_duration']
    shortest_rabi = params['rabi_pulse_initial_duration']
    
    fit_data, a, b, c, d = fit_rabi(avgs, a, b, c, d, num_patterns, longest_rabi)
    print("final parameters:")
    print("offset: ", a)
    print("amplitude: ", b)
    print("frequency: ", c*10**9)
    print("phi", d)
    
    
    x = np.linspace(shortest_rabi, longest_rabi, num_patterns)
    plt.plot(x, avgs, 'ko')
    plt.plot(x, fit_data, 'r')
    plt.xlabel("$t_{rabi}$ (ns)")
    plt.ylabel("V")
    plt.title(Path(fn).stem)
    plt.show()
    
    