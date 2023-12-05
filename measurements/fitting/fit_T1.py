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
import os
import json
import pickle as pkl
from matplotlib.widgets import Slider,Button,TextBox
import matplotlib.pylab as pylab
from IPython.display import display, clear_output
params = {'legend.fontsize': 'x-small',
          'figure.figsize': (15, 8),
         'axes.labelsize': 'x-small',
         'axes.titlesize':'x-small',
         'xtick.labelsize':'x-small',
         'ytick.labelsize':'x-small'}
pylab.rcParams.update(params)
def objective_T1(x, a, b, c):

	return a + (b*np.exp(-x/c))

def fit_T1(y_data, init_a, init_b, init_c, x_data):
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
    #pop = dp.get_population_v_pattern(arr, params['v_threshold'], flipped = False)
    pop = [np.average(i) for i in arr]

    #first get pattern avgs
    avgs = np.zeros(len(arr))
    for i in range(len(arr)):
        avgs[i] = np.average(arr[i])
        
    a = 300
    b = 100
    c = 100000
    #params = params[params['measurement']]
    longest_T1 = params['T1_final_gap']
    shortest_T1 = params['T1_init_gap']
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
    plt.xlabel("$t_{T1}$ (ns)")
    plt.ylabel("V")
    plt.title("T1 measurement")
    plt.show()

def fit_subax(ax, x, exp, fit_data, title):
    
    
    #for axis in ['top', 'bottom', 'left', 'right']:
    #    ax.spines[axis].set_linewidth(2.5)
    
    line, = ax.plot(x, exp, 'ko', markersize=5)
    line2, = ax.plot(x, fit_data[0], 'r', linewidth=2.5)
    ax.set_xlabel("$t_{T1}$ (ns)")
    ax.set_ylabel("V")
    ax.set_title(title)
    
    text = "offset: " + str(round(fit_data[1], 3)) + \
    "\n amp: " + str(round(fit_data[2], 3)) + \
    "\ntau: " + str(round(fit_data[3]/1000, 3)) + " us"

    textA = ax.text(.98, .98, text, fontsize = 10, horizontalalignment='right',
        verticalalignment='top', color='green',transform=ax.transAxes)
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
    #-----------------------------------------------cutting index------------------------------
    cut_index = 1
    new_avgs = []
    for i in range(len(avgs)):
        new_avgs.append(avgs[i][cut_index:])

    print(np.shape(new_avgs))
    #pop = dp.get_population_v_pattern(arr, params['v_threshold'], flipped = True)
    #pop = [np.average(i) for i in arr]

    #first get pattern avgs
    #avgs = np.zeros(len(arr))
    #for i in range(len(arr)):
    #    avgs[i] = np.average(arr[i])
        
    a = [np.average(new_avgs[0]),np.average(new_avgs[1]),np.average(new_avgs[2]),
        np.average(new_avgs[3]),np.average(new_avgs[4]),np.average(new_avgs[5])] #offset
    index = 0
    b = [new_avgs[0][index]-a[0],new_avgs[1][index]-a[1],new_avgs[2][index]-a[2],
         new_avgs[3][index]-a[3],new_avgs[4][index]-a[4],new_avgs[5][index]-a[5]]
    c = 5679

    if params['measurement'] == 'T1':
        print('T1')
        title = "T1"
        params = params['T1']
        longest_T1 = params['T1_final_gap']
        shortest_T1 = params['T1_init_gap']
    elif params['measurement'] == 'echo':
        print('echo')
        title = "Echo"
        params = params['echo']
        shortest_T1 = params['echo_initial_t']
        longest_T1 = params['echo_final_t']
    num_patterns = len(new_avgs[0])
    
    x = np.linspace(shortest_T1,longest_T1, num_patterns)
    
    fig, ax_array = plt.subplots(2,3)
    #widgets
    ax_slide = plt.axes([0.1,0.01,0.35,0.03])
    ax_box = plt.axes([0.55, 0.01, 0.15, 0.03])
    theta = Slider(ax_slide,"Theta [Deg]",valmin= 0, valmax = 360, valinit= 0, valstep= 0.1)
    textbox = TextBox(ax_box,'T1', initial='5679')
    #(pattern_avgs_cA, pattern_avgs_cA_sub, pattern_avgs_cB, pattern_avgs_cB_sub, mags, mags_sub)
    data_ans = fit_T1(new_avgs[0], a[0], b[0], c, x)
    data_as = fit_T1(new_avgs[1], a[1], b[1], c, x)
    data_bns = fit_T1(new_avgs[2], a[2], b[2], c, x)
    data_bs = fit_T1(new_avgs[3], a[3], b[3], c, x)
    data_mns = fit_T1(new_avgs[4], a[4], b[4], c, x)
    data_ms = fit_T1(new_avgs[5], a[5], b[5], c, x)
    
    #ms, ms_a, ms_b, ms_c
    plt.rcParams.update({'font.size': 22})
    lineE0,lineF0,text0 = fit_subax(ax_array[0,0], x, new_avgs[0], data_ans, "chA nosub")
    lineE1,lineF1,text1 = fit_subax(ax_array[1,0], x, new_avgs[1], data_as, "chA sub")
    lineE2,lineF2,text2 = fit_subax(ax_array[0,1], x, new_avgs[2], data_bns, "chB nosub")
    lineE3,lineF3,text3 = fit_subax(ax_array[1,1], x, new_avgs[3], data_bs, "chB sub")
    lineE4,lineF4,text4 = fit_subax(ax_array[0,2], x, new_avgs[4], data_mns, "Mags nosub")
    lineE5,lineF5,text5 = fit_subax(ax_array[1,2], x, new_avgs[5], data_ms, "mags sub")

    plt.suptitle('{} Measurement'.format(str(title)))
    #textbox function
    def update_freq_guess(text: str):
        update_fit(text,eval(text))
        return text

    #button function
    def update_fit(event,text=1/650):
        current_val = theta.val
        avgs = data.get_avgs(current_val)
        new_avgs = []
        for i in range(len(avgs)):
            new_avgs.append(avgs[i][cut_index:])
        #new fit 
        a = [np.average(new_avgs[0]),np.average(new_avgs[1]),np.average(new_avgs[2]),
            np.average(new_avgs[3]),np.average(new_avgs[4]),np.average(new_avgs[5])] #offset
        b = 3*[abs(max(new_avgs[0])-min(new_avgs[0])),abs(max(new_avgs[1])-min(new_avgs[1])),abs(max(new_avgs[2])-min(new_avgs[2])),
            abs(max(new_avgs[3])-min(new_avgs[3])),abs(max(new_avgs[4])-min(new_avgs[4])),abs(max(new_avgs[5])-min(new_avgs[5]))] #amp
        #guess for the update plot
        c = float(eval(textbox.text))  #freq
        af = np.zeros(len(a))
        bf = np.zeros(len(a))
        cf = np.zeros(len(a))
        df = np.zeros(len(a))
        data_ans, af[0], bf[0], cf[0] = fit_T1(new_avgs[0], a[0], b[0], c, x)
        data_as, af[1], bf[1], cf[1]= fit_T1(new_avgs[1], a[1], b[1], c, x)
        data_bns, af[2], bf[2], cf[2] = fit_T1(new_avgs[2], a[2], b[2], c, x)
        data_bs, af[3], bf[3], cf[3] = fit_T1(new_avgs[3], a[3], b[3], c, x)
        data_mns, af[4], bf[4], cf[4] = fit_T1(new_avgs[4], a[4], b[4], c, x)
        data_ms, af[5], bf[5], cf[5] = fit_T1(new_avgs[5], a[5], b[5], c, x)

        text=[]
        for i in range(len(a)):
            context = "offset: " + str(round(af[i], 3)) + \
                      "\n amp: " + str(round(bf[i], 3)) + \
                      "\ntau: " + str(round(cf[i]/1000, 3)) + " us"
            text.append(context)


        lineF0.set_ydata(data_ans)
        text0.set_text(text[0])
        delta = abs( min(avgs[0]) - max(avgs[0]))*0.05
        ax_array[0,0].set_ylim([min(avgs[0]) - delta, max(avgs[0]) +delta])

        lineF1.set_ydata(data_as)
        text1.set_text(text[1])
        delta = abs( min(avgs[1]) - max(avgs[1]))*0.05
        ax_array[1,0].set_ylim([min(avgs[1])- delta,max(avgs[1])+ delta])

        lineF2.set_ydata(data_bns)
        text2.set_text(text[2])
        delta = abs( min(avgs[2]) - max(avgs[2]))*0.05
        ax_array[0,1].set_ylim([min(avgs[2]) - delta ,max(avgs[2]) + delta])

        lineF3.set_ydata(data_bs)
        text3.set_text(text[3])
        delta = abs( min(avgs[3]) - max(avgs[3]))*0.05
        ax_array[1,1].set_ylim([min(avgs[3])- delta,max(avgs[3]) + delta])

        lineF4.set_ydata(data_mns)
        text4.set_text(text[4])
        delta = abs( min(avgs[4]) - max(avgs[4]))*0.05
        ax_array[0,2].set_ylim([min(avgs[4]) - delta ,max(avgs[4]) + delta])

        lineF5.set_ydata(data_ms)
        text5.set_text(text[5])
        delta = abs( min(avgs[5]) - max(avgs[5]))*0.05
        ax_array[1,2].set_ylim([min(avgs[5])- delta,max(avgs[5]) + delta])
        
        fig.canvas.draw_idle()
    #slider function
    def update_plot(val):
        current_val = theta.val
        avgs = data.get_avgs(current_val)
        new_avgs = []
        for i in range(len(avgs)):
            new_avgs.append(avgs[i][cut_index:])
        
        lineE0.set_ydata(new_avgs[0])
        delta = abs( min(avgs[0]) - max(avgs[0]))*0.05
        ax_array[0,0].set_ylim([min(avgs[0]) - delta, max(avgs[0]) +delta])

        lineE1.set_ydata(new_avgs[1])
        delta = abs( min(avgs[1]) - max(avgs[1]))*0.05
        ax_array[1,0].set_ylim([min(avgs[1])- delta,max(avgs[1])+ delta])

        lineE2.set_ydata(new_avgs[2])
        delta = abs( min(avgs[2]) - max(avgs[2]))*0.05
        ax_array[0,1].set_ylim([min(avgs[2]) - delta ,max(avgs[2]) + delta])

        lineE3.set_ydata(new_avgs[3])
        delta = abs( min(avgs[3]) - max(avgs[3]))*0.05
        ax_array[1,1].set_ylim([min(avgs[3])- delta,max(avgs[3]) + delta])

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
    
    
    
