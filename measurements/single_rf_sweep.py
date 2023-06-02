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
import json

if __name__ == "__main__":
    
    f = open('general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    
    

    #name = params['name']

    directory = tkf.askdirectory()
    name = directory + "/" + params['name'] + "_"
    decimation = params['decimation']
    
    j_file = open(name+"json.json", 'w')
    json.dump(params, j_file, indent = 4)
    j_file.close()

    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    wait_time = zero_length * zero_multiple
    
    pattern_repeat = params['pattern_repeat']
    seq_repeat = params['seq_repeat']
    
    p1start = params['p1start']
    p1stop = params['p1stop']
    p1step = params['p1step']


    p1 = params['p1']
    '''
    if p1 == 'wq' or p1 == 'wr':
        p1start = params['p1start']
        p1stop = params['p1stop']
        p1step = params['p1step']
    else:
        p1start = params['p1start']
        p1stop = params['p1stop']
        p1step = params['p1step']
        '''
    
    awg = be.get_awg()
    num_patterns = awg.get_seq_length()

    board = ats.Board(systemId = 1, boardId = 1)
    npt.ConfigureBoard(board)

    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    run_funcs.init_params(params)


    cAp_sub, cBp_sub, cAp_nosub, cBp_nosub, mags_sub, mags_nosub = run_funcs.single_sweep(name,
                                                                    awg,
                                                                    board,
                                                                    num_patterns,
                                                                    params,
                                                                    live_plot = False)
    
    np.save(name + "chA_sub", cAp_sub)
    np.save(name + "chB_sub", cBp_sub)
    np.save(name + "chA_nosub", cAp_nosub)
    np.save(name + "chB_nosub", cBp_nosub)
    np.save(name + "mags_sub", mags_sub)
    np.save(name + "mags_sub", mags_nosub)
    
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
    for i in range(num_patterns):
        plt.plot(x, mags_sub[i])
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
    for i in range(num_patterns):
       plt.plot(x, mags_nosub[i])
    plt.title('Magnitude nosub')
    
    plt.show()
    