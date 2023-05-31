# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:15:50 2023

@author: lqc
"""

import numpy as np
#import daq_programs
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from scipy.optimize import curve_fit
import json
import os
import sys
sys.path.append("../../")
from lib import data_process as dp

def objective_ramsey(x, a, b, t2, f, phi):
    
    return a + b*np.exp(-x/t2)*np.sin(2*np.pi*f*x + phi)


def fit_ramsey(y, init_a, init_b, init_t2, init_f, init_phi, num_points, max_length):
    x_data = np.linspace(0, max_length, num_points)
    y_data = y
    initial  = [init_a, init_b, init_t2, init_f, init_phi]
    popt, _ = curve_fit(objective_ramsey, x_data, y_data, p0 = initial)
    a, b, t2, f, phi = popt
    
    new_data = objective_ramsey(x_data, a, b, t2, f, phi)
    
    
    #plt.plot(x_data,objective_T1(x_data,a,b,c))
    #plt.plot(x_data,y_data)
    #plt.show()
    
    return new_data, a, b, t2, f, phi


if __name__ == "__main__":

    fn = askopenfilename()
    
    nf = '\\'.join(fn.split('/')[0:-1]) + "/"

    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f:
                with open(nf + f) as file:
                    params = json.load(file)

    arr = np.load(fn)
    #pop = dp.get_population_v_pattern(arr, params['v_threshold'], flipped = False)
    pop = [np.average(i) for i in arr]
    print(np.shape(arr), np.shape(pop))
    #first get pattern avgs
    avgs = np.zeros(len(arr))
    for i in range(len(arr)):
        avgs[i] = np.average(arr[i])
        
    a = 216.2
    b = .01
    t2 = 6000
    f = 1/3000
    phi = 1.57
    shortest_ramsey = params['ramsey_gap_1_init']
    longest_ramsey = params['ramsey_gap_1_final']
    num_patterns = len(arr)
    
    
    fit_data, new_a, new_b, new_t2, new_f, new_phi = fit_ramsey(pop, a, b, t2, f, phi, num_patterns, longest_ramsey)
    print("offset: ", new_a)
    print("amplitude: ", new_b)
    print("tau: ", new_t2)
    print("phi: ", new_phi)
    print("frequency: ", new_f)
    
    
    x = np.linspace(shortest_ramsey, longest_ramsey, num = num_patterns)
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.5)
    
    plt.plot(x, pop, 'ko', markersize=10)
    plt.plot(x, fit_data, 'r', linewidth=3.5)
    plt.xlabel("$t_{ramsey}$ (ns)")
    plt.ylabel("PE")
    plt.title("Ramsey measurement")
    plt.show()
    
    
    
    