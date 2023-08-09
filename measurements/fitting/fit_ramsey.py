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
from matplotlib.widgets import Slider,Button,TextBox

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

    params = params[params['measurement']]
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
    plt.ylabel("V")
    plt.title("Ramsey measurement")
    plt.show()


def fit_subax(ax, x, exp, fit_data, title):
    line, = ax.plot(x, exp, 'ko', markersize=10)
    line2, = ax.plot(x, fit_data[0], 'r', linewidth=3.5)
    ax.set_xlabel("$t_{Ramsey}$ (ns)")
    ax.set_ylabel("V")
    ax.set_title(title)
    
    text = "offset: " + str(round(fit_data[1], 3)) + \
    "\n amp: " + str(round(fit_data[2], 3)) + \
    "\nT2: " + str(round(fit_data[3]/1000, 3)) + " us" + \
    "\nfreq: " + str(round(fit_data[4], 10)) + " GHz" + \
    "\nphi: " + str(round(fit_data[2], 3))
    
    textA = ax.text(.98, .98, text, fontsize = 10,color = 'green', horizontalalignment='right',
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

    #arrs = data.get_data_arrs()
    avgs = data.get_avgs()
    #pop = dp.get_population_v_pattern(arr, params['v_threshold'], flipped = True)
    #pop = [np.average(i) for i in arr]

    #first get pattern avgs
    #avgs = np.zeros(len(arr))
        
    a = [np.average(avgs[0]),np.average(avgs[1]),np.average(avgs[2]),
        np.average(avgs[3]),np.average(avgs[4]),np.average(avgs[5])] #offset
    b = 3*[abs(max(avgs[0])-min(avgs[0])),abs(max(avgs[1])-min(avgs[1])),abs(max(avgs[2])-min(avgs[2])),
        abs(max(avgs[3])-min(avgs[3])),abs(max(avgs[4])-min(avgs[4])),abs(max(avgs[5])-min(avgs[5]))] #amp
    t2 = 3000
    f = 1/3800
    phi = 1.57
    params = params['ramsey']
    longest_T1 = params['ramsey_gap_1_final']
    shortest_T1 = params['ramsey_gap_1_init']
    num_patterns = len(avgs[0])
    
    x = np.linspace(shortest_T1,longest_T1, num_patterns)
    
    fig, ax_array = plt.subplots(2,3,figsize=(13, 7))
    
    #widgets
    ax_slide = plt.axes([0.1,0.01,0.35,0.03])
    ax_box = plt.axes([0.55, 0.01, 0.15, 0.03])
    theta = Slider(ax_slide,"Theta [Deg]",valmin= 0, valmax = 360, valinit= 0, valstep= 0.1)
    textbox = TextBox(ax_box,'Freq(GHz)', initial='1/3800')

    #(pattern_avgs_cA, pattern_avgs_cA_sub, pattern_avgs_cB, pattern_avgs_cB_sub, mags, mags_sub)
    fig.set_tight_layout(True)
    data_ans = fit_ramsey(avgs[0], a[0], b[0], t2 , f, phi, x)
    data_as = fit_ramsey(avgs[1], a[1], b[1], t2 , f, phi, x)
    data_bns = fit_ramsey(avgs[2], a[2], b[2], t2 , f, phi, x)
    data_bs = fit_ramsey(avgs[3], a[3], b[3], t2 , f, phi, x)
    data_mns = fit_ramsey(avgs[4], a[4], b[4], t2 , f, phi, x)
    data_ms = fit_ramsey(avgs[5], a[5], b[5], t2 , f, phi, x)
    
    plt.rcParams.update({'font.size': 22})
    lineE0,lineF0,text0 = fit_subax(ax_array[0,0], x, avgs[0], data_ans, "chA nosub")
    lineE1,lineF1,text1 = fit_subax(ax_array[1,0], x, avgs[1], data_as, "chA sub")
    lineE2,lineF2,text2 = fit_subax(ax_array[0,1], x, avgs[2], data_bns, "chB nosub")
    lineE3,lineF3,text3 = fit_subax(ax_array[1,1], x, avgs[3], data_bs, "chB sub")
    lineE4,lineF4,text4 = fit_subax(ax_array[0,2], x, avgs[4], data_mns, "mags nosub")
    lineE5,lineF5,text5 = fit_subax(ax_array[1,2], x, avgs[5], data_ms, "mags sub")
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.suptitle('Ramsey measurement')
    

    #textbox function
    def update_freq_guess(text: str):
        update_fit(text,eval(text))
        return text

    #button function
    def update_fit(event,text=1/650):
        current_val = theta.val
        avgs = data.get_avgs(current_val)
        #new fit 
        a = [np.average(avgs[0]),np.average(avgs[1]),np.average(avgs[2]),
            np.average(avgs[3]),np.average(avgs[4]),np.average(avgs[5])] #offset
        b = 3*[abs(max(avgs[0])-min(avgs[0])),abs(max(avgs[1])-min(avgs[1])),abs(max(avgs[2])-min(avgs[2])),
            abs(max(avgs[3])-min(avgs[3])),abs(max(avgs[4])-min(avgs[4])),abs(max(avgs[5])-min(avgs[5]))] #amp
        #guess for the update plot
        t2 = 3000.
        f = float(eval(textbox.text))  #freq
        d = np.pi/2 #phase

        af = np.zeros(len(a))
        bf = np.zeros(len(a))
        t2f = np.zeros(len(a))
        ff = np.zeros(len(a))
        df = np.zeros(len(a))
        #new_data, a, b, t2, f, phi
        data_ans, af[0], bf[0], t2f[0], ff[0], df[0] = fit_ramsey(avgs[0], a[0], b[0], t2 , f, d, x)
        data_as, af[1], bf[1], t2f[1], ff[1], df[1] = fit_ramsey(avgs[1], a[1], b[1], t2, f, d, x)
        data_bns, af[2], bf[2], t2f[2], ff[2], df[2] = fit_ramsey(avgs[2], a[2], b[2], t2, f, d, x)
        data_bs, af[3], bf[3], t2f[3], ff[3], df[3] = fit_ramsey(avgs[3], a[3], b[3], t2, f, d, x)
        data_mns, af[4], bf[4], t2f[4], ff[4], df[4] = fit_ramsey(avgs[4], a[4], b[4], t2, f, d, x)
        data_ms, af[5], bf[5], t2f[5], ff[5], df[5] = fit_ramsey(avgs[5], a[5], b[5], t2, f, d, x)
        text=[]
        for i in range(len(a)):
            context = "offset: " + str(round(af[i], 3)) + \
            "\n amp: " + str(round(bf[i], 3)) + \
            "\nT2: " + str(round(t2f[i]/1000, 3)) + " us" + \
            "\nfreq: " + str(round(ff[i], 10)) + " GHz" + \
            "\nphi: " + str(round(df[i], 3))
            text.append(context)


        lineF0.set_ydata(data_ans)
        text0.set_text(text[0])
        ax_array[0,0].set_ylim([min(avgs[0]),max(avgs[0])])

        lineF1.set_ydata(data_as)
        text1.set_text(text[1])
        ax_array[1,0].set_ylim([min(avgs[1]),max(avgs[1])])

        lineF2.set_ydata(data_bns)
        text2.set_text(text[2])
        ax_array[0,1].set_ylim([min(avgs[2]),max(avgs[2])])

        lineF3.set_ydata(data_bs)
        text3.set_text(text[3])
        ax_array[1,1].set_ylim([min(avgs[3]),max(avgs[3])])

        lineF4.set_ydata(data_mns)
        text4.set_text(text[4])
        ax_array[0,2].set_ylim([min(avgs[4]),max(avgs[4])])

        lineF5.set_ydata(data_ms)
        text5.set_text(text[5])
        ax_array[1,2].set_ylim([min(avgs[5]),max(avgs[5])])
        fig.canvas.draw_idle()
    #slider function
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

        #lineE4.set_ydata(avgs[4])
        #ax_array[0,2].set_ylim([min(avgs[4]),max(avgs[4])])

        #lineE5.set_ydata(avgs[5])
        #ax_array[1,2].set_ylim([min(avgs[5]),max(avgs[5])])

        update_fit(val,textbox.text)
        fig.canvas.draw_idle()

    #assign the functions when acting on it
    theta.on_changed(update_plot)
    textbox.on_submit(update_freq_guess)



    plt.show()


if __name__ == "__main__":
    #legacy_fit()
    new_fit()
    