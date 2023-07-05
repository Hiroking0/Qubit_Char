# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:00:50 2021

@author: Crow108
"""

import numpy as np
#import daq_programs
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from scipy.optimize import curve_fit
import sys
sys.path.append("../../")
sys.path.append("../")
import os
import json
import pickle as pkl

def objective_T1(x, a, b, c):

	return a + (b*np.exp(-x/c))

def fit_T1(y, init_a, init_b, init_c, x_data):
    #x_data = np.linspace(0, max_length, num_points)
    y_data = y
    initial  = [init_a,init_b,init_c]
    popt, _ = curve_fit(objective_T1, x_data, y_data, p0 = initial)
    a, b, c= popt
    
    new_data = objective_T1(x_data,a,b,c)
    
    return new_data, a, b, c


def legacy_fit():
    fn = askopenfilename()
    
    nf = '\\'.join(fn.split('/')[0:-1]) + "/"

    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f:
                with open(nf + f) as file:
                    params = json.load(file)

    arr = np.load(fn)
    #pop = dp.get_population_v_pattern(arr, params['v_threshold'], flipped = True)
    pop = [np.average(i) for i in arr]

    #first get pattern avgs
    avgs = np.zeros(len(arr))
    for i in range(len(arr)):
        avgs[i] = np.average(arr[i])
        
    a = 207.5
    b = .1
    c = 5679

    longest_T1 = params['echo_final_t']
    shortest_T1 = params['echo_initial_t']
    num_patterns = len(arr)
    
    fit_data, new_a, new_b, new_c = fit_T1(pop, a, b, c, num_patterns, longest_T1)
    print("final data:")
    print("offset: ", new_a)
    print("amplitude: ", new_b)
    print("tau: ", new_c/1000)
    
    x = np.linspace(shortest_T1,longest_T1, num_patterns)
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.5)
    plt.plot(x, pop, 'ko', markersize=10)
    plt.plot(x, fit_data, 'r', linewidth=3.5)
    plt.xlabel("$t_{echo}$ (ns)")
    plt.ylabel("V")
    plt.title("echo measurement")

    plt.show()
    
    
def fit_subax(ax, x, exp, fit_data, title):
    
    
    #for axis in ['top', 'bottom', 'left', 'right']:
    #    ax.spines[axis].set_linewidth(2.5)
    
    ax.plot(x, exp, 'ko', markersize=10)
    ax.plot(x, fit_data[0], 'r', linewidth=3.5)
    ax.set_xlabel("$t_{echo}$ (ns)")
    ax.set_ylabel("V")
    ax.set_title(title)
    text = "offset: " + str(round(fit_data[1], 3)) + "\n amp: " + str(round(fit_data[2], 3)) + "\ntau: " + str(round(fit_data[3]/1000, 3)) + " us"
    ax.text(.98, .98, text, fontsize = 10, horizontalalignment='right',
        verticalalignment='top', transform=ax.transAxes)
    
    
def new_fit():
    fn = askopenfilename(filetypes=[("Pickles", "*.pkl")])
    
    with open(fn, 'rb') as pickled_file:
        data = pkl.load(pickled_file)
    
    nf = '\\'.join(fn.split('/')[0:-1]) + "/"
    no_ext_file = ''.join(fn.split('/')[-1])[:-4]
    
    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f and no_ext_file in f:
                with open(nf + f) as file:
                    params = json.load(file)

    #arrs = data.get_data_arrs()
    avgs = data.get_avgs()
    #pop = dp.get_population_v_pattern(arr, params['v_threshold'], flipped = True)
    #pop = [np.average(i) for i in arr]

    #first get pattern avgs
    #avgs = np.zeros(len(arr))
    #for i in range(len(arr)):
    #    avgs[i] = np.average(arr[i])
        
    a = 207.5
    b = .1
    c = 5679
    params = params['echo']
    longest_T1 = params['echo_final_t']
    shortest_T1 = params['echo_initial_t']
    num_patterns = len(avgs[0])
    
    x = np.linspace(shortest_T1,longest_T1, num_patterns)
    
    fig, ax_array = plt.subplots(2,3)
    
    #ans, bns, mns, as, bs, ms
    data_ans = fit_T1(avgs[0], a, b, c, x)
    data_bns = fit_T1(avgs[1], a, b, c, x)
    data_mns = fit_T1(avgs[2], a, b, c, x)
    data_as = fit_T1(avgs[3], a, b, c, x)
    data_bs = fit_T1(avgs[4], a, b, c, x)
    data_ms = fit_T1(avgs[5], a, b, c, x)
    
    #ms, ms_a, ms_b, ms_c
    plt.rcParams.update({'font.size': 22})
    fit_subax(ax_array.flatten()[0], x, avgs[0], data_ans, "chA nosub")
    fit_subax(ax_array.flatten()[1], x, avgs[1], data_bns, "chB nosub")
    fit_subax(ax_array.flatten()[2], x, avgs[2], data_mns, "Mags nosub")
    fit_subax(ax_array.flatten()[3], x, avgs[3], data_as, "chA sub")
    fit_subax(ax_array.flatten()[4], x, avgs[4], data_bs, "chB sub")
    fit_subax(ax_array.flatten()[5], x, avgs[5], data_ms, "mags sub")

    plt.suptitle('Echo measurement')
    plt.show()



if __name__ == "__main__":
    
    #legacy_fit()
    new_fit()
    
    
