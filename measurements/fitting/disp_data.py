# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:44:38 2023

@author: lqc
"""
from tkinter.filedialog import askopenfilename
import numpy as np
import sys
sys.path.append("../../")
from lib import data_process as dp
import matplotlib.pyplot as plt
import os
import json
import pandas
import fit_rabi

def disp_sequence():
    fn = askopenfilename()
    nf = '\\'.join(fn.split('/')[0:-1]) + "/"
    plt.rcParams.update({'font.size': 18})
    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f:
                with open(nf + f) as file:
                    params = json.load(file)

    #try to open file as numpy array, this means it was many patterns.
    
    arr = np.load(fn)
    #print(np.shape(arr))
    pop = dp.get_population_v_pattern(arr, params['v_threshold'])
    print(pop)
    dp.plot_histogram(arr)
    params = params[params['measurement']]
    x = []
    if "rabi" in nf.lower():
        x = np.linspace(params['rabi_pulse_initial_duration'], params['rabi_pulse_end_duration'], num = len(arr))
    if "ramsey" in nf.lower():
        x = np.linspace(params['ramsey_gap_1_init'], params['ramsey_gap_1_final'], num = len(arr))
    if "t1" in nf.lower():
        x = np.linspace(params['T1_init_gap'], params['T1_final_gap'], num = len(arr))
    if "echo" in nf.lower():
        x = np.linspace(params['echo_initial_t'], params['echo_final_t'], num = len(arr))
    if len(arr) > 1:
        avgs = [np.average(a) for a in arr]
        plt.plot(avgs)
        plt.show()
    '''
    if len(arr) > 1:
        pop = dp.get_population_v_pattern(arr, params['v_threshold'], flipped = True)
        plt.plot(pop)
        #print(pop)
        plt.show()
     '''  
    #n_points = params['seq_repeat'] * params['pattern_repeat']
    wq = params['set_wq']*(10**9)
    kb = 1.38e-23
    hbar = 1.054e-34
    del_E = (-hbar * 2 * np.pi * wq)
    
    denom = kb * np.log((1-pop[1])/(pop[1]))
    
    T = -del_E/denom
    print("Effective tempurature (mK):", T*(10**3))
    

def disp_single_sweep():
    
    file = askopenfilename()
    csvFile = pandas.read_csv(file, sep = ',', engine = 'python')
    num_patterns = max(csvFile['pattern_num']) + 1
    sweep_num = int(len(csvFile['pattern_num'])/num_patterns)
    shape = (num_patterns, sweep_num)
    
    #plt.rcParams.update({'font.size': 80})
    sweep_param = csvFile.columns[-1]
    min_sweep = min(csvFile[sweep_param].to_list())
    max_sweep = max(csvFile[sweep_param].to_list())
    print(min_sweep, max_sweep)
    #x = range(num_patterns)
    x = np.linspace(0, 2000, num = 51)
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


    plt.rcParams.update({'font.size': 16})
    plt.subplot(2,3,5)
    chB_sub = csvFile['chB_sub'].to_list()
    chB_sub = np.reshape(chB_sub, shape)
    chB_sub = np.transpose(chB_sub)
    plt.pcolormesh(x, y, chB_sub, shading = 'auto')
    plt.title("ChB_sub")
    plt.xlabel("$t_{ramsey} (ns)$", fontsize=16)
    plt.ylabel('qubit drive frequency (GHz)', fontsize=16)
    
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

def disp_3_chevrons():
    fn = askopenfilename()
    
    nf = '\\'.join(fn.split('/')[0:-1]) + "/"

    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f:
                with open(nf + f) as file:
                    params = json.load(file)
    
    plt.rcParams.update({'font.size': 18})
    csvFile = pandas.read_csv(fn, sep = ',', engine = 'python')
    num_patterns = max(csvFile['pattern_num']) + 1
    sweep_num = int(len(csvFile['pattern_num'])/num_patterns)
    shape = (num_patterns, sweep_num)
    
    sweep_vals = csvFile['wq'].to_list()
    sweep_vals = np.reshape(sweep_vals, shape)
    sweep_vals = np.transpose(sweep_vals)
    
    mags_nosub = csvFile['mag_nosub'].to_list()
    mags_nosub = np.reshape(mags_nosub, shape)
    mags_nosub = np.transpose(mags_nosub)
    #print(np.shape(mags_nosub))
    #pop16 = dp.get_population_v_pattern(mags_nosub, params['v_threshold'], flipped = False)
    
    
    a = 305.65
    b = .2
    c = 1/200
    d = np.pi/2
    num_patterns = len(mags_nosub[0])
    longest_rabi = 600
    shortest_rabi = 0
    x = np.linspace(shortest_rabi, longest_rabi, num = num_patterns)
    f1, a, b, c, d = fit_rabi.fit_rabi(mags_nosub[16], a, b, c, d, num_patterns, longest_rabi)
    plt.plot(x, f1, 'b')
    #plt.plot(x, mags_nosub[16], 'bo')
    
    a = 305.65
    b = .1
    c = 1/144
    d = np.pi/2
    f2, a, b, c, d = fit_rabi.fit_rabi(mags_nosub[20], a, b, c, d, num_patterns, longest_rabi)
    plt.plot(x, f2, 'r')
    #plt.plot(x, mags_nosub[20], 'ro')
    
    a = 305.67
    b = .08
    c = 1/100
    d = -np.pi/2
    f3, a, b, c, d = fit_rabi.fit_rabi(mags_nosub[24], a, b, c, d, num_patterns, longest_rabi)
    plt.plot(x, f3, 'k')
    #plt.plot(x, mags_nosub[24], 'ko')
    
    
    plt.xlabel('t_$rabi$ (ns)')
    plt.ylabel("V")
    center_freq = 3.2388e9
    #deviation from center freq legend
    
    l1 = str((sweep_vals[16][0]-center_freq)/1e6)
    l2 = str((sweep_vals[20][0]-center_freq)/1e6)
    l3 = str((sweep_vals[24][0]-center_freq)/1e6)
    
    plt.legend([ "+" +l1 + "MHz", "+" +l2 + "MHz", "+" +l3 + "MHz"])
    #plt.legend([sweep_vals[16][0], sweep_vals[20][0], sweep_vals[24][0]])
    plt.title("Three chevron slices")
    plt.show()


def show_sweep_output():
    fn = askopenfilename()
    
    nf = '\\'.join(fn.split('/')[0:-1]) + "/"

    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f:
                with open(nf + f) as file:
                    params = json.load(file)
    #plt.rcParams.update({'font.size': 18})
    csvFile = pandas.read_csv(fn, sep = ',', engine = 'python')
    num_patterns = max(csvFile['pattern_num']) + 1
    sweep_num = int(len(csvFile['pattern_num'])/num_patterns)
    shape = (num_patterns, sweep_num)
    x = csvFile[csvFile.columns[-1]].to_list()
    x = x[:int(len(x)/num_patterns)]
    
    print(np.shape(x))
    mags_nosub = csvFile['mag_nosub'].to_list()
    mags_nosub = np.reshape(mags_nosub, shape)
    
    
    mags_sub = csvFile['mag_sub'].to_list()
    mags_sub = np.reshape(mags_sub, shape)
    
    cAp_nosub = csvFile['chA_nosub'].to_list()
    cAp_nosub = np.reshape(cAp_nosub, shape)
    
    cAp_sub = csvFile['chA_sub'].to_list()
    cAp_sub = np.reshape(cAp_sub, shape)
    
    
    cBp_nosub = csvFile['chB_nosub'].to_list()
    cBp_nosub = np.reshape(cBp_nosub, shape)
    
    cBp_sub = csvFile['chB_sub'].to_list()
    cBp_sub = np.reshape(cBp_sub, shape)
    print(np.shape(mags_sub))
    
    
    plt.subplot(2,3,1)
    for i in range(num_patterns):
        plt.plot(x, cAp_sub[i])
    plt.title('channel a')
    
    plt.subplot(2,3,2)
    for i in range(num_patterns):
        plt.plot(x, cBp_sub[i])
    plt.xlabel("$w_{drive}$ (GHz)")
    plt.ylabel("V")
    plt.title('Channel B')
    plt.legend(["no pulse", "pi pulse"])
    
    plt.subplot(2,3,3)
    for j in range(num_patterns):
        plt.plot(x, mags_sub[j])
    plt.title('Magnitude sub')
    
    plt.subplot(2,3,4)
    for i in range(num_patterns):
        plt.plot(x, cAp_nosub[i])
    plt.title('channel a nosub')
    
    plt.subplot(2,3,5)
    for i in range(num_patterns):
        plt.plot(x, cBp_nosub[i])
    
    plt.title('channel b nosub')
    
    plt.subplot(2,3,6)
    for j in range(num_patterns):
        plt.plot(x, mags_nosub[j])
    plt.title('Magnitude nosub')
    
    plt.show()


def disp_double_sweep():
    pass



if __name__ == "__main__":
    #disp_double_sweep()
    disp_sequence()
    #show_sweep_output()
    #disp_single_sweep()
    #disp_3_chevrons()
    
    
    