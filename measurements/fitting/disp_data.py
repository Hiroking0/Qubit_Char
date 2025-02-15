# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:44:38 2023

@author: lqc
"""

import tkinter as tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory
import numpy as np
import sys
sys.path.append("../../")
from lib import data_process as dp
import matplotlib.pyplot as plt
import os
import json
import pandas
import fit_rabi
import pickle as pkl
from fit_rabi import  fit_rabi
from scipy.optimize import curve_fit
from multiprocessing import Process, Manager
from qcodes.dataset import (
    Measurement,
    initialise_or_create_database_at,
    load_by_run_spec,
    load_from_netcdf,
    load_or_create_experiment,
    plot_dataset,)

def disp_sequence():
    fn = askopenfilename(filetypes=[("Pickles", "*.pkl")])
    nf = '\\'.join(fn.split('/')[0:-1]) + "/" #Gets the path of the file and adds a /
    no_ext_file = ''.join(fn.split('/')[-1])[:-4]
    
    
    plt.rcParams.update({'font.size': 18})

    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f and no_ext_file in f:
                with open(nf + f) as file:
                    params = json.load(file)

    with open(fn, 'rb') as pickled_file:
        data = pkl.load(pickled_file)
    #data is a Data_arrs type

    if params['measurement'] == 'readout' or params['measurement'] == 'npp':
        timestep = 1
    else:
        timestep = params[params['measurement']]['step']

    #This code is for adding a phase offset to nosub or sub arrays
    


    
    (a_nosub, a_sub, b_nosub, b_sub, mags_nosub, mags_sub, readout_A, readout_B) = data.get_data_arrs()

    complex_arr = np.zeros((len(a_nosub), len(a_nosub[0])), dtype=np.complex_)
    complex_arr_sub = np.zeros((len(a_nosub), len(a_nosub[0])), dtype=np.complex_)
    angle_arr = np.angle(complex_arr_sub.flatten())
    theta = np.average(angle_arr)
    theta = np.radians(10)
    exp = np.exp(1j*theta)

    print("THETA", theta)
    for i in range(len(a_nosub)):
        for j in range(len(a_nosub[0])):
            t_i = a_nosub[i,j]
            t_q = b_nosub[i,j]
            t_new = np.multiply(t_i+1j*t_q, exp)
            complex_arr[i,j] = t_new

            t_i_sub = a_sub[i,j]
            t_q_sub = b_sub[i,j]
            t_new_sub = np.multiply(t_i_sub+1j*t_q_sub, exp)
            complex_arr_sub[i,j] = t_new_sub


    new_a_nosub = np.real(complex_arr)
    new_b_nosub = np.imag(complex_arr)
    


    new_a_sub = np.real(complex_arr_sub)
    new_b_sub = np.imag(complex_arr_sub)

    setattr(data, 'a_nosub', new_a_nosub)
    setattr(data, 'b_nosub', new_b_nosub)

    setattr(data, 'a_sub', new_a_sub)
    setattr(data, 'b_sub', new_b_sub)
    dp.plot_np_file(data, timestep)


    '''
    #print(np.shape(arr))
    pop = dp.get_population_v_pattern(arr, params['v_threshold'])
    print(pop)
    dp.plot_histogram(arr)
    params = params[params['measurement']]
    x = []
    measurement = params['measurement']

    if measurement == "rabi":
        x = np.linspace(params['rabi_pulse_initial_duration'], params['rabi_pulse_end_duration'], num = len(arr))
    if measurement == "ramsey":
        x = np.linspace(params['ramsey_gap_1_init'], params['ramsey_gap_1_final'], num = len(arr))
    if measurement == "t1":
        x = np.linspace(params['T1_init_gap'], params['T1_final_gap'], num = len(arr))
    if measurement == "echo" or measurement == "echo_1ax":
        x = np.linspace(params['echo_initial_t'], params['echo_final_t'], num = len(arr))


    #n_points = params['seq_repeat'] * params['pattern_repeat']
    wq = params['set_wq']*(10**9)
    kb = 1.38e-23
    hbar = 1.054e-34
    del_E = (-hbar * 2 * np.pi * wq)
    
    denom = kb * np.log((1-pop[1])/(pop[1]))
    
    T = -del_E/denom
    print("Effective tempurature (mK):", T*(10**3))
    '''


def plot_mesh_subax(ax, x, y, data, title, xlabel, ylabel):
    ax.pcolormesh(x, y, data, shading = 'auto')
    ax.set_title(title)
    ax.set_xlabel('pattern #')
    ax.set_ylabel(ylabel)

def disp_single_sweep():
    
    file = askopenfilename()


    nf = '\\'.join(file.split('/')[0:-1]) + "/"

    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f:
                with open(nf + f) as t_file:
                    params = json.load(t_file)


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
    y = np.arange(params['p1start'], params['p1stop'], params['p1step'])
    #x = np.linspace(0, 2000, num = 51)
    x = np.linspace(0, num_patterns, num = 51)
    #print(sweep_num)

    fig, ax_array = plt.subplots(2,3)
    ax_array = ax_array.flatten()
    chA_nosub = csvFile['chA_nosub'].to_list()
    chA_nosub = np.reshape(chA_nosub, shape)
    chA_nosub = np.transpose(chA_nosub)
    plot_mesh_subax(ax_array[0], x, y, chA_nosub, "chA_nosub", "pattern #", sweep_param)

    chB_nosub = csvFile['chB_nosub'].to_list()
    chB_nosub = np.reshape(chB_nosub, shape)
    chB_nosub = np.transpose(chB_nosub)
    plot_mesh_subax(ax_array[1], x, y, chB_nosub, "chB_nosub", "pattern #", sweep_param)
    
    mags_nosub = csvFile['mag_nosub'].to_list()
    mags_nosub = np.reshape(mags_nosub, shape)
    mags_nosub = np.transpose(mags_nosub)
    plot_mesh_subax(ax_array[2], x, y, mags_nosub, "mags_nosub", "pattern #", sweep_param)

    chA_sub = csvFile['chA_sub'].to_list()
    chA_sub = np.reshape(chA_sub, shape)
    chA_sub = np.transpose(chA_sub)
    plot_mesh_subax(ax_array[3], x, y, chA_sub, "chA_sub", "pattern #", sweep_param)

    chB_sub = csvFile['chB_sub'].to_list()
    chB_sub = np.reshape(chB_sub, shape)
    chB_sub = np.transpose(chB_sub)
    plot_mesh_subax(ax_array[4], x, y, chB_sub, "chB_sub", "pattern #", sweep_param)
    
    mags_sub = csvFile['mag_sub'].to_list()
    mags_sub = np.reshape(mags_sub, shape)
    mags_sub = np.transpose(mags_sub)
    plot_mesh_subax(ax_array[5], x, y, mags_sub, "mags_sub", "pattern #", sweep_param)
    
    
    plt.show()


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
    
    fns = askopenfilenames()
    
    nf = '\\'.join(fns[0].split('/')[0:-1]) + "/"

    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f:
                with open(nf + f) as file:
                    params = json.load(file)
    
    y = np.arange(params['p1start'], params['p1stop'], params['p1step'])
    x = np.arange(params['p2start'], params['p2stop'], params['p2step'])
    z = np.zeros((len(y), len(x)))
    #index 4 is b nosub
    for f in fns:
        csvFile = pandas.read_csv(f, sep = ',', engine = 'python')
        temp_z = csvFile['chA_nosub'].to_list()
        temp_x = csvFile[csvFile.columns[-1]][0]
        temp_x = np.where(x == temp_x)[0][0]
        z[:, temp_x] = temp_z
        
    plt.pcolormesh(x, y, z)
    plt.show()

def get_temp_thresh():
    fn = askopenfilename(filetypes=[("Pickles", "*.pkl")])
    nf = '\\'.join(fn.split('/')[0:-1]) + "/" #Gets the path of the file and adds a /
    no_ext_file = ''.join(fn.split('/')[-1])[:-4]
    
    
    #plt.rcParams.update({'font.size': 18})

    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f and no_ext_file in f:
                with open(nf + f) as file:
                    params = json.load(file)

    with open(fn, 'rb') as pickled_file:
        data = pkl.load(pickled_file)




    #data is a Data_arrs type

    if params['measurement'] == 'readout' or params['measurement'] == 'npp':
        timestep = 1
    else:
        timestep = params[params['measurement']]['step']


    #dp.plot_np_file(data, timestep)

    
    #print(np.shape(arr))
    #ans, bns, mns, as, bs, ms
    #ans, as, bns, bs, mns, ms
    manager = Manager()
    acproc = Process(target = dp.plot_np_file, args = (data, timestep,))
    acproc.start()

    #---------------------------------------------------------------------channel-----------------------------------------------
    channel_name = input("Enter channel:")
    thresh = input("Enter threshold:")
    thresh = eval(str(thresh))

    print('________Results________')

    if channel_name == 'chA_nosub':
        channel = 0
    elif channel_name == 'chA_sub':
        channel = 1
    elif channel_name == 'chB_nosub':
        channel = 2
    elif channel_name == 'chB_sub':
        channel = 3
    elif channel_name == 'mags_nosub':
        channel = 4
    elif channel_name == 'mags_sub':
        channel = 5
    else:
        print('Error')
    
    #n_points = params['seq_repeat'] * params['pattern_repeat']
    wq = (params['set_wq'] + params[params['measurement']]['ssb_freq'])*1e9
    print('wq',wq)
    kb = 1.38649e-23
    hbar = 1.05457e-34
    del_E = (-hbar * 2 * np.pi * wq)
    
    print('')
    print('### Temp with user thresh ###')
    dp.eff_temp(data.get_data_arrs()[channel],thresh,wq)

    #fitting
    #data is a Data_arrs type
    #(a_nosub, a_sub, b_nosub, b_sub, mags_nosub, mags_sub, readout_A, readout_B) = data.get_data_arrs() 

    def cuts(tb,data,thresh = thresh):
        freq_G, bin_edges = np.histogram(data[0],200,range=(np.min(data),np.max(data)))
        freq_E, bin_edges = np.histogram(data[1],200,range=(np.min(data),np.max(data)))
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        if tb == 'b':
            cut = (bin_centers<thresh)

            cutdataG = freq_G[cut]
            cutdataE = freq_E[cut]

            binG = bin_centers[cut]
            binE = bin_centers[cut]
        elif tb == 't':
            cut = (bin_centers>thresh)

            cutdataG = freq_G[cut]
            cutdataE = freq_E[cut]

            binG = bin_centers[cut]
            binE = bin_centers[cut]
        #return freq_G,freq_E, bin_centers
        return cutdataG,binG,cutdataE,binE,bin_centers

    def gaus(x,peak,center,dev):
        return peak*np.exp(-(x-center)**2/(2*dev**2))
    
    def _2gaussian(x, amp1,cen1,sigma1, amp2,cen2,sigma2,area):
        return (amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2))))
    
    #extract cuts
    cutdataG1,binG1,cutdataE1,binE1,bin_centers = cuts('b',data.get_data_arrs()[channel])
    cutdataG2,binG2,cutdataE2,binE2,bin_centers = cuts('t',data.get_data_arrs()[channel])
    #freq_G,freq_E, bin_centers = cuts('b',b_nosub)
    

    #----------------------------------intial guess-------------------------------------------------------------------
    p0 =[np.max(cutdataG2),np.average(binG2),0.2*(max(binG2)-min(binG2))]
    #p0n = [300,214.044,0.1,600,214.705,0.1,abs(np.sum(freq_G))]
    
    #Fits with cuts
    popt, pcov = curve_fit(gaus,binG2,cutdataG2,p0)
    pars1 = popt
    
    popt, pcov = curve_fit(gaus,binE1,cutdataE1,p0)
    pars2 = popt
    
    #fits using double gaus
    #popt1, pcov = curve_fit(_2gaussian, bin_centers, freq_G, p0n)
    #amp1,cen1,sigma1, amp2,cen2,sigma2,freq = popt1
    #pars_1 = popt1[0:3]
    #pars_2 = popt1[3:6]

    '''popt2, pcov = curve_fit(_2gaussian, bin_centers, freq_E, p0n)
    amp1,cen1,sigma1, amp2,cen2,sigma2 = popt
    pars_1 = popt2[0:3]
    pars_2 = popt2[3:6]'''

    #calculate temp using fit
    #eric
    nthresh = np.average([pars1[1],pars2[1]])
    #eff temp func
    print('')
    print('### Temp with fit thresh ###')
    dp.eff_temp(data.get_data_arrs()[channel],nthresh,wq)

    #plotting
    plt.hist(data.get_data_arrs()[channel][0],bins=200,histtype='step',label='0')
    plt.hist(data.get_data_arrs()[channel][1],bins=200,histtype='step',label='1')
    plt.plot(bin_centers,gaus(bin_centers,*pars1),label='curve fit')
    plt.plot(bin_centers,gaus(bin_centers,*pars2),label='curve fit')
    plt.plot([thresh,thresh],[0,pars2[0]],'b--',label='guess:{:.3f}'.format(thresh))
    plt.plot([nthresh,nthresh],[0,pars2[0]],'r-',label='fit:{:.3f}'.format(nthresh))
    plt.title(channel_name)
    #plt.plot(binG,gaus(binG,600,214.052,.1),label='fit')

    #plt.plot(binG,cutdataG2,label='this')
    #plt.plot(binE,cutdataE)
    #print(gaus(x,peak_fit,center_fit,dev_fit))
    #print(p0n)
    #print(popt1)
    print('')
    print('________End________')
    '''plt.plot(bin_centers, gaus(bin_centers,*pars_1))
    plt.plot(bin_centers, gaus(bin_centers,*pars_2))
    plt.plot(bin_centers,_2gaussian(bin_centers,*popt1),label='fit')
    plt.plot(bin_centers,_2gaussian(bin_centers,*p0n),label='guess')'''
    plt.legend()
    plt.show()
    


    
    
def two_rpm():
    fn = askopenfilename(filetypes=[("Pickles", "*.pkl")])
    fn2 = askopenfilename(filetypes=[("Pickles", "*.pkl")])
    with open(fn, 'rb') as pickled_file:
        data = pkl.load(pickled_file)
    with open(fn2, 'rb') as pickled_file:
        data2 = pkl.load(pickled_file)
    
    nf = '\\'.join(fn.split('/')[0:-1]) + "/"
    no_ext_file = ''.join(fn.split('/')[-1])[:-4]
    
    for (root, dirs, files) in os.walk(nf):
        for f in files:
            if ".json" in f and no_ext_file in f:
                with open(nf + f) as file:
                    params = json.load(file)

    #arrs = data.get_data_arrs()
    avgs = data.get_avgs()
    avgs2 = data2.get_avgs()

        
    a = 226.6 #offset
    b = .2 #amp
    c = 1/250   #freq
    d = np.pi/2 #phase
    params = params['rabi']
    longest_T1 = params['rabi_pulse_initial_duration']
    shortest_T1 = params['rabi_pulse_end_duration']
    num_patterns = len(avgs[0])
    
    x = np.linspace(shortest_T1,longest_T1, num_patterns)
    
    
    #pattern_avgs_cA, pattern_avgs_cA_sub, pattern_avgs_cB, pattern_avgs_cB_sub, mags, mags_sub
    #data_ans = fit_rabi(avgs[0], a, b, c, d, x)
    data_bns = fit_rabi(avgs[2], a, b, c, d, x)
    data_bns2 = fit_rabi(avgs2[2], a, b, c, d, x)
    #data_mns = fit_rabi(avgs[2], a, b, c, d, x)
    #data_as = fit_rabi(avgs[3], a, b, c, d, x)
    #data_bs = fit_rabi(avgs[4], a, b, c, d, x)
    #data_ms = fit_rabi(avgs[5], a, b, c, d, x)
    
    #ms, ms_a, ms_b, ms_c
    fig, ax = plt.subplots(1,1)
    plt.rcParams.update({'font.size': 22})
    #fit_subax(ax_array.flatten()[0], x, avgs[0], data_ans, "chA nosub")
    #fit_subax(ax_array, x, avgs[1], data_bns, "chB nosub")
    #fit_subax(ax_array, x, avgs2[1], data_bns2, "chB nosub")
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.5)

    ax.plot(x, avgs[2], 'ko', markersize=10)
    ax.plot(x, data_bns[0], 'r', linewidth=3.5)
    ax.set_xlabel("$t_{Rabi}$ (ns)")
    ax.set_ylabel("V")

    ax.plot(x, avgs2[2], 'bo', markersize=10)
    ax.plot(x, data_bns2[0], 'y', linewidth=3.5)
    ax.set_xlabel("$t_{Rabi}$ (ns)")
    ax.set_ylabel("V")

    #ax.set_title(title)
    #text = "offset: " + str(round(fit_data[1], 3)) + \
    #        "\n amp: " + str(round(fit_data[2], 3)) + \
    #        "\nfreq: " + str(round(fit_data[3], 10)) + " GHz" + \
    #        "\nphase: "+ str(round(fit_data[4], 3))

    #ax.text(.98, .98, text, fontsize = 10, horizontalalignment='right',
    #    verticalalignment='top', transform=ax.transAxes)

    #fit_subax(ax_array.flatten()[2], x, avgs[2], data_mns, "Mags nosub")
    #fit_subax(ax_array.flatten()[3], x, avgs[3], data_as, "chA sub")
    #fit_subax(ax_array.flatten()[4], x, avgs[4], data_bs, "chB sub")
    #fit_subax(ax_array.flatten()[5], x, avgs[5], data_ms, "mags sub")

    plt.suptitle('RPM measurement')
    plt.legend(['pi preperation data', 'pi prep fit', '20% pi prep data', '20% pi prep fit'])
    plt.show()
def continousmeas():
    file_path = askopenfilename()
    data = load_from_netcdf(file_path)
    type(data)
    plot_dataset(data)
    plt.show()

def night_run():
    datas=[]
    def plot_data(fn):
        nf = '\\'.join(fn.split('/')[0:-1]) + "/" #Gets the path of the file and adds a /
        no_ext_file = ''.join(fn.split('/')[-1])[:-4]


        plt.rcParams.update({'font.size': 18})

        with open(fn, 'rb') as pickled_file:
            #print(pickled_file)
            data = pkl.load(pickled_file)
            #data is a Data_arrs type
        return data


    # Create a Tkinter root window (it will not be shown)
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Ask the user to select a folder
    folder_path = askdirectory(title="Select Folder")
    print(folder_path)

    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if  f.endswith(".pkl") and os.path.isfile(os.path.join(folder_path, f))]
    timestamp=[]
    # Iterate through the files and plot each one
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        timestamp.append(file_name[-10:-4])
        fn=file_name
        print(file_name)
        nf = folder_path + "/" #Gets the path of the file and adds a /
        no_ext_file = ''.join(fn.split('/')[-1])[:-4]


        plt.rcParams.update({'font.size': 18})

        for (root, dirs, files) in os.walk(nf):
            for f in files:
                if ".json" in f and no_ext_file in f:
                    with open(nf + f) as file:
                        params = json.load(file)
        data = plot_data(file_path)
        datas.append(data)
    dp.plot_nightrun(datas,params,timestamp)
    dp.night_run_color(datas,params,timestamp)
    plt.show()
    # Show the plot



if __name__ == "__main__":
    #get_temp_thresh()
    #disp_double_sweep()
    #disp_sequence()
    #disp_nightrun()
    #show_sweep_output() #each pattern will be overlayed on each other
    #disp_single_sweep() #3d plot pattern # is x axis
    #disp_3_chevrons()
    #two_rpm()
    continousmeas()
    #night_run()
    