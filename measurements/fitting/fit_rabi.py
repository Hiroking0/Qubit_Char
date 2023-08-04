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
import json
import os
import sys
sys.path.append("../../")
from lib import data_process as dp
import pickle as pkl
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

def objective_rabi(x, a, b, c, d):
	return a + (b*np.sin(2*np.pi*c*x+d))

def fit_rabi(y_data, init_a, init_b, init_c, init_d, x_data):
    initial  = [init_a, init_b, init_c, init_d]
    popt, _ = curve_fit(objective_rabi, x_data, y_data, p0 = initial)
    a, b , c, d = popt
    new_data = objective_rabi(x_data,a,b,c,d)
    
    return new_data, a, b, c, d


def legacy_fit():
    f = open('../general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    
    fn = askopenfilename()
    
    nf = '\\'.join(fn.split('/')[0:-1]) + "/"

    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f:
                with open(nf + f) as file:
                    params = json.load(file)


    '''
    fn2 = askopenfilename()
     
    nf2 = '\\'.join(fn.split('/')[0:-1]) + "/"
    
    for (root, dirs, files) in os.walk(nf2):
        for f in files:
            if ".json" in f:
                with open(nf2 + f) as file:
                    params = json.load(file)
    arr2 = np.load(fn2)
    avgs2 = np.zeros(len(arr2))
    for i in range(len(arr2)):
        avgs2[i] = np.average(arr2[i])
        '''

    arr = np.load(fn)
    pop = dp.get_population_v_pattern(arr, params['v_threshold'], flipped = False)
    #first get pattern avgs
    avgs = np.zeros(len(arr))
    for i in range(len(arr)):
        avgs[i] = np.average(arr[i])
        
    a = 292.3
    b = .2
    c = 1/240
    d = np.pi/2
    num_patterns = len(arr)
    params = params[params['measurement']]
    longest_rabi = params['rabi_pulse_end_duration']
    shortest_rabi = params['rabi_pulse_initial_duration']
    
    fit_data, a, b, c, d = fit_rabi(avgs, a, b, c, d, num_patterns, longest_rabi)
    print("final parameters I:")
    print("offset: ", a)
    print("amplitude: ", b)
    print("frequency: ", c*10**9)
    print("phi", d)
    
    '''
    fit_data2, a, b, c, d = fit_rabi(avgs2, a, b, c, d, num_patterns, longest_rabi)
    print("final parameters Q:")
    print("offset: ", a)
    print("amplitude: ", b)
    print("frequency: ", c*10**9)
    print("phi", d)
    '''
    
    
    x = np.linspace(shortest_rabi, longest_rabi, num_patterns)
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.5)
    
    
    plt.plot(x, avgs, 'bo', markersize=10)
    #plt.plot(x, avgs2, 'k*', markersize=10)
    plt.plot(x, fit_data, 'r', linewidth=3.5)
    #plt.plot(x, fit_data2, 'r', linewidth=3.5)
    plt.xlabel("$t_{rabi}$ (ns)")
    plt.ylabel("V")
    #plt.legend(["I rotation", "Q rotation"])
   # plt.title("rabi measurement")
    plt.show()

def fit_subax(ax, x, exp, fit_data, title,line=0):
    
    
    #for axis in ['top', 'bottom', 'left', 'right']:
    #    ax.spines[axis].set_linewidth(2.5)
    
    line, = ax.plot(x, exp, 'ko', markersize=10)
    line2, = ax.plot(x, fit_data[0], 'r', linewidth=3.5)
    ax.set_xlabel("$t_{Rabi}$ (ns)")
    ax.set_ylabel("V")
    ax.set_title(title)
    text = "offset: " + str(round(fit_data[1], 3)) + \
            "\n amp: " + str(round(fit_data[2], 3)) + \
            "\nfreq: " + str(round(fit_data[3], 10)) + " GHz" + \
            "\nphase: "+ str(round(fit_data[4], 3))

    ax.text(.98, .98, text, fontsize = 10, horizontalalignment='right',
        verticalalignment='top', transform=ax.transAxes)
    return line,line2


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



    if params['measurement'] == 'readout' or params['measurement'] == 'npp':
        timestep = 1
    else:
        timestep = params[params['measurement']]['step']


    # arrs = data.get_data_arrs()
    avgs = data.get_avgs()
    #pop = dp.get_population_v_pattern(arr, params['v_threshold'], flipped = True)
    #pop = [np.average(i) for i in arr]

    #first get pattern avgs
    #avgs = np.zeros(len(arr))
    #for i in range(len(arr)):
    #    avgs[i] = np.average(arr[i])
        
    a = 210 #offset
    b = .0075 #amp
    c = 1/500  #freq
    d = np.pi/2 #phase
    params = params['rabi']
    shortest_T1 = params['rabi_pulse_initial_duration']
    longest_T1 = params['rabi_pulse_end_duration']
    num_patterns = len(avgs[0])
    
    x = np.linspace(shortest_T1,longest_T1, num_patterns)
    print(shortest_T1)
    
    fig, ax_array = plt.subplots(2,3,figsize=(13, 7))
    y=-0.4
    ax_slide = plt.axes([0.1,0.01,0.35,0.03])
    ax_button = plt.axes([0.5, 0.01, 0.1, 0.03])
    theta = Slider(ax_slide,"Theta",valmin= 0, valmax = 360, valinit= 0, valstep= 0.1)
    update_button = Button(ax_button,"Update Fit",hovercolor = 'green')

    #(pattern_avgs_cA, pattern_avgs_cA_sub, pattern_avgs_cB, pattern_avgs_cB_sub, mags, mags_sub)
    data_ans = fit_rabi(avgs[0], a, b, c, d, x)
    data_as = fit_rabi(avgs[1], a, b, c, d, x)
    data_bns = fit_rabi(avgs[2], a, b, c, d, x)
    data_bs = fit_rabi(avgs[3], a, b, c, d, x)
    data_mns = fit_rabi(avgs[4], a, b, c, d, x)
    data_ms = fit_rabi(avgs[5], a, b, c, d, x)
    
    #ms, ms_a, ms_b, ms_c
    plt.rcParams.update({'font.size': 15})
    lineE0,lineF0 = fit_subax(ax_array[0,0], x, avgs[0], data_ans, "chA nosub")
    lineE1,lineF1 = fit_subax(ax_array[1,0], x, avgs[1], data_as, "chA sub")
    lineE2,lineF2 = fit_subax(ax_array[0,1], x, avgs[2], data_bns, "chB nosub")
    lineE3,lineF3 = fit_subax(ax_array[1,1], x, avgs[3], data_bs, "chB sub")
    lineE4,lineF4 = fit_subax(ax_array[0,2], x, avgs[4], data_mns, "mags nosub")
    lineE5,lineF5 = fit_subax(ax_array[1,2], x, avgs[5], data_ms, "mags sub")
    

    plt.suptitle('Rabi measurement with and shift {} deg'.format(0))
    
    def update_plot(val):
        current_val = theta.val
        avgs = data.get_avgs(current_val)
        
        lineE0.set_ydata(avgs[0])
        ax_array[0,0].set_ylim([min(avgs[0]),max(avgs[0])])

        lineE1.set_ydata(avgs[1])
        ax_array[1,0].set_ylim([min(avgs[1]),max(avgs[1])])

        lineE2.set_ydata(avgs[2])
        ax_array[0,1].set_ylim([min(avgs[2]),max(avgs[2])])

        lineE3.set_ydata(avgs[3])
        ax_array[1,1].set_ylim([min(avgs[3]),max(avgs[3])])

        lineE4.set_ydata(avgs[4])
        ax_array[0,2].set_ylim([min(avgs[4]),max(avgs[4])])

        lineE5.set_ydata(avgs[5])
        ax_array[1,2].set_ylim([min(avgs[5]),max(avgs[5])])



        fig.canvas.draw_idle()

    def update_fit(event):
        #new fit 
        data_ans = fit_rabi(avgs[0], a, b, c, d, x)[0]
        data_as = fit_rabi(avgs[1], a, b, c, d, x)[0]
        data_bns = fit_rabi(avgs[2], a, b, c, d, x)[0]
        data_bs = fit_rabi(avgs[3], a, b, c, d, x)[0]
        data_mns = fit_rabi(avgs[4], a, b, c, d, x)[0]
        data_ms = fit_rabi(avgs[5], a, b, c, d, x)[0]
        
        lineF0.set_ydata(data_ans)
        ax_array[0,0].set_ylim([min(avgs[0]),max(avgs[0])])

        lineF1.set_ydata(data_as)
        ax_array[1,0].set_ylim([min(avgs[1]),max(avgs[1])])

        lineF2.set_ydata(data_bns)
        ax_array[0,1].set_ylim([min(avgs[2]),max(avgs[2])])

        lineF3.set_ydata(data_bs)
        ax_array[1,1].set_ylim([min(avgs[3]),max(avgs[3])])

        lineF4.set_ydata(data_mns)
        ax_array[0,2].set_ylim([min(avgs[4]),max(avgs[4])])

        lineF5.set_ydata(data_ms)
        ax_array[1,2].set_ylim([min(avgs[5]),max(avgs[5])])
        fig.canvas.draw_idle()



    update_button.on_clicked(update_fit)
    theta.on_changed(update_plot)

    plt.show()



if __name__ == "__main__":
    
    #legacy_fit()
    new_fit()
    
    
    
    
    