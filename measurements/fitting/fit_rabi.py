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
from matplotlib.widgets import Slider,Button,TextBox

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

def fit_subax(ax, x, exp, fit_data, title):
    
    
    #for axis in ['top', 'bottom', 'left', 'right']:
    #    ax.spines[axis].set_linewidth(2.5)
    
    line, = ax.plot(x, exp, 'ko', markersize=5)
    line2, = ax.plot(x, fit_data[0], 'r', linewidth=2.5)
    ax.set_xlabel("$t_{Rabi}$ (ns)")
    ax.set_ylabel("V")
    ax.set_title(title)
    text = "offset: " + str(round(fit_data[1], 3)) + \
            "\n amp: " + str(round(fit_data[2], 3)) + \
            '\n pi: ' + str(round(1/(fit_data[3]*2), 2)) + " ns" + \
            "\nphase: "+ str(round(fit_data[4], 3))

    textA = ax.text(.98, .98, text, fontsize = 10,color='green', horizontalalignment='right',
        verticalalignment='top', transform=ax.transAxes)
    return line,line2,textA


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


    # arrs = data.get_data_arrs()
    avgs = data.get_avgs()
    #pop = dp.get_population_v_pattern(arr, params['v_threshold'], flipped = True)
    #pop = [np.average(i) for i in arr]

    #first get pattern avgs
    #avgs = np.zeros(len(arr))
    #for i in range(len(arr)):
    #    avgs[i] = np.average(arr[i])
    
    #guess for the intial plot    
    a = [np.average(avgs[0]),np.average(avgs[1]),np.average(avgs[2]),
        np.average(avgs[3]),np.average(avgs[4]),np.average(avgs[5])] #offset
    b = 3*[abs(max(avgs[0])-min(avgs[0])),abs(max(avgs[1])-min(avgs[1])),abs(max(avgs[2])-min(avgs[2])),
        abs(max(avgs[3])-min(avgs[3])),abs(max(avgs[4])-min(avgs[4])),abs(max(avgs[5])-min(avgs[5]))] #amp
    c = 1/150  #freq
    d = np.pi/2 #phase
    if params['measurement'] == 'rabi':
        print('rabi')
        title = "Rabi"
        params = params['rabi']
        shortest_T1 = params['rabi_pulse_initial_duration']
        longest_T1 = params['rabi_pulse_end_duration']
    elif params['measurement'] == 'effect_temp':
        print('effect_temp')
        title = "Effective Temperature"
        params = params['effect_temp']
        shortest_T1 = params['rabi_start']
        longest_T1 = params['rabi_stop']

    num_patterns = len(avgs[0])
    
    x = np.linspace(shortest_T1,longest_T1, num_patterns)

    
    fig, ax_array = plt.subplots(2,3,figsize=(13, 7))

    #widgets
    ax_slide = plt.axes([0.1,0.01,0.35,0.03])
    ax_box = plt.axes([0.55, 0.01, 0.15, 0.03])
    ax_button_update = plt.axes([0.75, 0.01, 0.04, 0.03])
    ax_button_hide = plt.axes([0.79, 0.01, 0.04, 0.03])
    theta = Slider(ax_slide,"Theta [Deg]",valmin= 0, valmax = 360, valinit= 0, valstep= 0.1)
    textbox = TextBox(ax_box,'Freq(GHz)', initial='1/150')
    update = Button(ax_button_update,"Update",hovercolor = 'green')
    hide = Button(ax_button_hide,"Hide",hovercolor = 'red')

    #(pattern_avgs_cA, pattern_avgs_cA_sub, pattern_avgs_cB, pattern_avgs_cB_sub, mags, mags_sub)
    data_ans = fit_rabi(avgs[0], a[0], b[0], c, d, x)
    data_as = fit_rabi(avgs[1], a[1], b[1], c, d, x)
    data_bns = fit_rabi(avgs[2], a[2], b[2], c, d, x)
    data_bs = fit_rabi(avgs[3], a[3], b[3], c, d, x)
    data_mns = fit_rabi(avgs[4], a[4], b[4], c, d, x)
    data_ms = fit_rabi(avgs[5], a[5], b[5], c, d, x)
    
    #ms, ms_a, ms_b, ms_c
    plt.rcParams.update({'font.size': 15})
    lineE0,lineF0,text0 = fit_subax(ax_array[0,0], x, avgs[0], data_ans, "chA nosub")
    lineE1,lineF1,text1 = fit_subax(ax_array[1,0], x, avgs[1], data_as, "chA sub")
    lineE2,lineF2,text2 = fit_subax(ax_array[0,1], x, avgs[2], data_bns, "chB nosub")
    lineE3,lineF3,text3 = fit_subax(ax_array[1,1], x, avgs[3], data_bs, "chB sub")
    lineE4,lineF4,text4 = fit_subax(ax_array[0,2], x, avgs[4], data_mns, "mags nosub")
    lineE5,lineF5,text5 = fit_subax(ax_array[1,2], x, avgs[5], data_ms, "mags sub")
    

    plt.suptitle('{} Measurement'.format(str(title)))
    

    #textbox function
    def update_freq_guess(text: str):
        update_fit(text)
        return text

    #button function
    def update_fit(event):
        text=textbox.text
        current_val = theta.val
        avgs = data.get_avgs(current_val)
        #new fit 
        a = [np.average(avgs[0]),np.average(avgs[1]),np.average(avgs[2]),
            np.average(avgs[3]),np.average(avgs[4]),np.average(avgs[5])] #offset
        b = 3*[abs(max(avgs[0])-min(avgs[0])),abs(max(avgs[1])-min(avgs[1])),abs(max(avgs[2])-min(avgs[2])),
            abs(max(avgs[3])-min(avgs[3])),abs(max(avgs[4])-min(avgs[4])),abs(max(avgs[5])-min(avgs[5]))] #amp
        #guess for the update plot
        c = float(eval(textbox.text))  #freq
        d = np.pi/2 #phase
        af = np.zeros(len(a))
        bf = np.zeros(len(a))
        cf = np.zeros(len(a))
        df = np.zeros(len(a))
        data_ans, af[0], bf[0], cf[0], df[0] = fit_rabi(avgs[0], a[0], b[0], c, d, x)
        data_as, af[1], bf[1], cf[1], df[1] = fit_rabi(avgs[1], a[1], b[1], c, d, x)
        data_bns, af[2], bf[2], cf[2], df[2] = fit_rabi(avgs[2], a[2], b[2], c, d, x)
        data_bs, af[3], bf[3], cf[3], df[3] = fit_rabi(avgs[3], a[3], b[3], c, d, x)
        data_mns, af[4], bf[4], cf[4], df[4] = fit_rabi(avgs[4], a[4], b[4], c, d, x)
        data_ms, af[5], bf[5], cf[5], df[5] = fit_rabi(avgs[5], a[5], b[5], c, d, x)

        text=[]
        for i in range(len(a)):
            context = "offset: " + str(round(af[i], 3)) + \
                "\n amp: " + str(round(bf[i], 6)) + \
                '\n pi: ' +str(round(1/(cf[i]*2), 2)) + " ns" + \
                "\nphase: "+ str(round(df[i], 3))
            text.append(context)


        lineF0.set_data(x,data_ans)
        text0.set_text(text[0])
        delta = abs( min(avgs[0]) - max(avgs[0]))*0.05
        ax_array[0,0].set_ylim([min(avgs[0]) - delta, max(avgs[0]) +delta])

        lineF1.set_data(x,data_as)
        text1.set_text(text[1])
        delta = abs( min(avgs[1]) - max(avgs[1]))*0.05
        ax_array[1,0].set_ylim([min(avgs[1])- delta,max(avgs[1])+ delta])

        lineF2.set_data(x,data_bns)
        text2.set_text(text[2])
        delta = abs( min(avgs[2]) - max(avgs[2]))*0.05
        ax_array[0,1].set_ylim([min(avgs[2]) - delta ,max(avgs[2]) + delta])

        lineF3.set_data(x,data_bs)
        text3.set_text(text[3])
        delta = abs( min(avgs[3]) - max(avgs[3]))*0.05
        ax_array[1,1].set_ylim([min(avgs[3])- delta,max(avgs[3]) + delta])

        lineF4.set_data(x,data_mns)
        text4.set_text(text[4])
        delta = abs( min(avgs[4]) - max(avgs[4]))*0.05
        ax_array[0,2].set_ylim([min(avgs[4]) - delta ,max(avgs[4]) + delta])

        lineF5.set_data(x,data_ms)
        text5.set_text(text[5])
        delta = abs( min(avgs[5]) - max(avgs[5]))*0.05
        ax_array[1,2].set_ylim([min(avgs[5])- delta,max(avgs[5]) + delta])
        fig.canvas.draw_idle()
    #slider function
    def update_plot(val):
        current_val = theta.val
        avgs = data.get_avgs(current_val)
        
        lineE0.set_ydata(avgs[0])
        lineE0.set_xdata(x)
        delta = abs( min(avgs[0]) - max(avgs[0]))*0.05
        ax_array[0,0].set_ylim([min(avgs[0]) - delta, max(avgs[0]) +delta])

        lineE1.set_ydata(avgs[1])
        delta = abs( min(avgs[1]) - max(avgs[1]))*0.05
        ax_array[1,0].set_ylim([min(avgs[1])- delta,max(avgs[1])+ delta])

        lineE2.set_ydata(avgs[2])
        delta = abs( min(avgs[2]) - max(avgs[2]))*0.05
        ax_array[0,1].set_ylim([min(avgs[2]) - delta ,max(avgs[2]) + delta])

        lineE3.set_ydata(avgs[3])
        delta = abs( min(avgs[3]) - max(avgs[3]))*0.05
        ax_array[1,1].set_ylim([min(avgs[3])- delta,max(avgs[3]) + delta])

        #lineE4.set_ydata(avgs[4])
        #ax_array[0,2].set_ylim([min(avgs[4]),max(avgs[4])])

        #lineE5.set_ydata(avgs[5])
        #ax_array[1,2].set_ylim([min(avgs[5]),max(avgs[5])])

        #update_fit(val)
        fig.canvas.draw_idle()
    def clear_plot(val):
        lineF0.set_data([],[])
        lineF1.set_data([],[])
        lineF2.set_data([],[])
        lineF3.set_data([],[])
        lineF4.set_data([],[])
        lineF5.set_data([],[])
        fig.canvas.draw_idle()

    #assign the functions when acting on it
    theta.on_changed(update_plot)
    theta.on_changed(update_fit)
    textbox.on_submit(update_freq_guess)
    update.on_clicked(update_fit)
    hide.on_clicked(clear_plot)
    plt.show()



if __name__ == "__main__":
    
    #legacy_fit()
    new_fit()
    
    
    
    
    