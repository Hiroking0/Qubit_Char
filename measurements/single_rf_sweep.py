# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:38:30 2022

@author: lqc
"""

import sys
sys.path.append("../")
from lib import run_funcs
import numpy as np
import matplotlib.pyplot as plt
from lib import wave_construction as be
from instruments.alazar import ATS9870_NPT as npt
from instruments.alazar import atsapi as ats
import yaml
import time
import tkinter.filedialog as tkf

if __name__ == "__main__":
    
    f = open('general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()

    #name = params['name']

    directory = tkf.askdirectory()
    name = directory + "/" + params['name'] + "_"
    decimation = params['decimation']


    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    wait_time = zero_length * zero_multiple
    
    readout_start = params['readout_start']
    readout = params['readout_duration']
    pattern_repeat = params['pattern_repeat']
    seq_repeat = params['seq_repeat']
    
    acq_multiples = params['acq_multiples']
    samples_per_ac = 256*acq_multiples #length of acquisition in nS must be n*256
    
    GHz=1e9
    MHz=1e6
    kHz=1e3
    

    p1 = params['p1']
    if p1 == 'wq' or p1 == 'wr':
        p1start = params['p1start']*GHz
        p1stop = params['p1stop']*GHz
        p1step = params['p1step']*GHz
    else:
        p1start = params['p1start']
        p1stop = params['p1stop']
        p1step = params['p1step']

    
    awg = be.get_awg()
    
    num_patterns = awg.get_seq_length()
    
    
    for i in range(num_patterns):
        awg.set_seq_element_loop_cnt(i+1, pattern_repeat)
        time.sleep(.005)

    board = ats.Board(systemId = 1, boardId = 1)
    npt.ConfigureBoard(board)

    wlen = readout_start + readout + wait_time


    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    run_funcs.init_params(params)


    cAp_sub, cBp_sub, cAp_nosub, cBp_nosub = run_funcs.single_sweep(name, awg, board, num_patterns, params, p1, p1start, p1stop, p1step, params['avg_start'], params['avg_length'], live_plot = False)
    x = np.arange(p1start, p1stop, p1step)
    
    print(np.shape(cAp_sub))
    #plt.figure()
    plt.subplot(2,3,1)
    for i in range(num_patterns):
        plt.plot(x, cAp_sub[i])
    plt.title('channel a')
    
    plt.subplot(2,3,2)
    for i in range(num_patterns):
        plt.plot(x, cBp_sub[i])
    plt.title('channel b')
    
    
    plt.subplot(2,3,3)
    for j in range(num_patterns):
        mag_arr = [ np.sqrt(cAp_sub[j][i]**2 + cBp_sub[j][i]**2) for i in range(len(cAp_sub[0])) ]
        plt.plot(x, mag_arr)
    plt.title('Magnitude')
    
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
        mag_arr = [ np.sqrt(cAp_nosub[j][i]**2 + cBp_nosub[j][i]**2) for i in range(len(cAp_nosub[0])) ]
        plt.plot(x, mag_arr)
    plt.title('Magnitude nosub')
    
    
    
    #plt.figure()
    #plt.plot(x, ap1-ap2)
    #plt.title('difference between chA p1 and p2')
    
    plt.show()
    