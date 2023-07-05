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
import pickle as pkl
import sys
sys.path.append("../../")

def objective_ramsey(x, a, b, t2, f, phi):
    
    return a + b*np.exp(-x/t2)*np.sin(2*np.pi*f*x + phi)


def fit_ramsey(y_data, init_a, init_b, init_t2, init_f, init_phi, x_data):
    initial  = [init_a, init_b, init_t2, init_f, init_phi]
    popt, _ = curve_fit(objective_ramsey, x_data, y_data, p0 = initial)
    a, b, t2, f, phi = popt
    
    new_data = objective_ramsey(x_data, a, b, t2, f, phi)
    
    
    #plt.plot(x_data,objective_T1(x_data,a,b,c))
    #plt.plot(x_data,y_data)
    #plt.show()
    
    return new_data, a, b, t2, f, phi

def legacy_fit():
    fn = askopenfilename()
    
    nf = '\\'.join(fn.split('/')[0:-1]) + "/"

    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f:
                with open(nf + f) as file:
                    params = json.load(file)

    params = params['effect_temp']
    arr = np.load(fn)
    #pop = dp.get_population_v_pattern(arr, params['v_threshold'], flipped = False)
    pop = [np.average(i) for i in arr]
    print(np.shape(arr), np.shape(pop))
    #first get pattern avgs
    avgs = np.zeros(len(arr))
    for i in range(len(arr)):
        avgs[i] = np.average(arr[i])
        
    a = 223.8
    b = .1
    t2 = 3000
    f = 1/3800
    phi = 1.57
    shortest_ramsey = params['rabi_start']
    longest_ramsey = params['rabi_stop']
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
    plt.xlabel("$t_{Rabi}$ (ns)")
    plt.ylabel("V")
    plt.title("Effect Temp measurement")
    plt.show()


def fit_subax(ax, x, exp, fit_data, title):
    
    ax.plot(x, exp, 'ko', markersize=10)
    ax.plot(x, fit_data[0], 'r', linewidth=3.5)
    ax.set_xlabel("$t_{Rabi}$ (ns)")
    ax.set_ylabel("V")
    ax.set_title(title)
    text = "offset: " + str(round(fit_data[1], 3)) + \
    "\n amp: " + str(round(fit_data[2], 3)) + \
    "\nT2: " + str(round(fit_data[3]/1000, 3)) + " us" + \
    "\nfreq: " + str(round(fit_data[4], 10)) + " GHz" + \
    "\nphi: " + str(round(fit_data[2], 3))
    
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
        
    a = 223.8
    b = .1
    t2 = 3000
    f = 1/3800
    phi = 1.57
    params = params['ramsey']
    longest_T1 = params['ramsey_gap_1_final']
    shortest_T1 = params['ramsey_gap_1_init']
    num_patterns = len(avgs[0])
    
    x = np.linspace(shortest_T1,longest_T1, num_patterns)
    
    fig, ax_array = plt.subplots(2,3)
    
    #ans, bns, mns, as, bs, ms
    data_ans = fit_ramsey(avgs[0], a, b, t2, f, phi, x)
    data_bns = fit_ramsey(avgs[1], a, b, t2, f, phi, x)
    data_mns = fit_ramsey(avgs[2], a, b, t2, f, phi, x)
    data_as = fit_ramsey(avgs[3], a, b, t2, f, phi, x)
    data_bs = fit_ramsey(avgs[4], a, b, t2, f, phi, x)
    data_ms = fit_ramsey(avgs[5], a, b, t2, f, phi, x)
    
    plt.rcParams.update({'font.size': 22})
    fit_subax(ax_array.flatten()[0], x, avgs[0], data_ans, "chA nosub")
    fit_subax(ax_array.flatten()[1], x, avgs[1], data_bns, "chB nosub")
    fit_subax(ax_array.flatten()[2], x, avgs[2], data_mns, "Mags nosub")
    fit_subax(ax_array.flatten()[3], x, avgs[3], data_as, "chA sub")
    fit_subax(ax_array.flatten()[4], x, avgs[4], data_bs, "chB sub")
    fit_subax(ax_array.flatten()[5], x, avgs[5], data_ms, "mags sub")

    plt.suptitle('RPM')
    plt.show()


if __name__ == "__main__":
    #legacy_fit()
    new_fit()
    