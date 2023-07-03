# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:38:30 2022

@author: lqc
"""
import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from lib import wave_construction as be
from lib import run_funcs

from instruments.alazar import ATS9870_NPT as npt
from instruments.alazar import atsapi as ats

import tkinter.filedialog as tkf
import yaml
import json
def int_eval(data):
    return eval(str(data))
if __name__ == "__main__":
    
    f = open('general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    
    directory = tkf.askdirectory()
    name = directory + "/" + params['name'] + "_"
    
    j_file = open(name+"json.json", 'w')
    json.dump(params, j_file, indent = 4)
    j_file.close()
    
    #start_time is the time between triggering the AWG and start of the qubit pulse
    
    pattern_repeat = int_eval(params['pattern_repeat'])
    seq_repeat = int_eval(params['seq_repeat'])
    
    p1 = params['p1']
    p2 = params['p2']

    #sweep power J7201B
    decimation = int_eval(params['decimation'])
    
    awg = be.get_awg()
    
    num_patterns = awg.get_seq_length()
    
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    run_funcs.init_params(params)

    board = ats.Board(systemId = 1, boardId = 1)
    npt.ConfigureBoard(board)

    #returns arrays of channel A and B, averages of averages
    #shape is [num_patterns][p1 sweep length][p2 sweep length]
    f_A_nosub, f_B_nosub, f_A_sub, f_B_sub, f_M_sub, f_M_nosub = run_funcs.double_sweep(name,
                                                                                          awg,
                                                                                          board,
                                                                                          params,
                                                                                          num_patterns)
    
    np.save(name + "chA_sub", f_A_sub)
    np.save(name + "chB_sub", f_B_sub)
    np.save(name + "chA_nosub", f_A_nosub)
    np.save(name + "chB_nosub", f_B_nosub)
    np.save(name + "mags_sub", f_M_sub)
    np.save(name + "mags_sub", f_M_nosub)
    
    plt.figure()
    
    y = np.arange(int_eval(params['p1start']), int_eval(params['p1stop']), int_eval(params['p1step']))
    x = np.arange(int_eval(params['p2start']), int_eval(params['p2stop']), int_eval(params['p2step']))
    
    plt.subplot(2,3,1)
    for pattern in f_A_nosub:
        plt.pcolormesh(x, y, pattern)
    plt.xlabel(p2)
    plt.ylabel(p1)
    plt.title('ch A nosub')
    
    plt.subplot(2,3,2)
    for pattern in f_B_nosub:
        plt.pcolormesh(x, y, pattern)
    plt.xlabel(p2)
    plt.ylabel(p1)
    plt.title('ch B nosub')
    
    plt.subplot(2,3,3)
    for pattern in f_M_nosub:
        plt.pcolormesh(x, y, pattern)
    plt.xlabel(p2)
    plt.ylabel(p1)
    plt.title('mags nosub')
    

    plt.subplot(2,3,4)
    for pattern in f_A_sub:
        plt.pcolormesh(x, y, pattern)
    plt.xlabel(p2)
    plt.ylabel(p1)
    plt.title('ch A sub')
    
    plt.subplot(2,3,5)
    for pattern in f_B_sub:
        plt.pcolormesh(x, y, pattern)
    plt.xlabel(p2)
    plt.ylabel(p1)
    plt.title('ch B sub')
    
    plt.subplot(2,3,6)
    for pattern in f_M_sub:
        plt.pcolormesh(x, y, pattern)
    plt.xlabel(p2)
    plt.ylabel(p1)
    plt.title('mags sub')
    plt.suptitle("pattern 1")
    

    
    if len(f_M_sub[0] == 2):
        
        plt.figure()
        plt.title("pattern 2")
        plt.subplot(2,3,1)
        plt.pcolormesh(x, y, f_A_nosub[1])
        plt.xlabel(p2)
        plt.ylabel(p1)
        plt.title('ch A nosub')
        
        plt.subplot(2,3,2)
        plt.pcolormesh(x, y, f_B_nosub[1])
        plt.xlabel(p2)
        plt.ylabel(p1)
        plt.title('ch B nosub')
        
        plt.subplot(2,3,3)
        plt.pcolormesh(x, y, f_M_nosub[1])
        plt.xlabel(p2)
        plt.ylabel(p1)
        plt.title('mags nosub')
        

        plt.subplot(2,3,4)
        plt.pcolormesh(x, y, f_A_sub[1])
        plt.xlabel(p2)
        plt.ylabel(p1)
        plt.title('ch A sub')
        
        plt.subplot(2,3,5)
        plt.pcolormesh(x, y, f_B_sub[1])
        plt.xlabel(p2)
        plt.ylabel(p1)
        plt.title('ch B sub')
        
        plt.subplot(2,3,6)
        plt.pcolormesh(x, y, f_M_sub[1])
        plt.xlabel(p2)
        plt.ylabel(p1)
        plt.title('mags sub')
        plt.suptitle("pattern 2")
        
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.pcolormesh(x, y, np.abs(f_B_nosub[1]-f_B_nosub[0]))
        plt.xlabel(p2)
        plt.ylabel(p1)
        plt.title('difference B')
        
        plt.subplot(2,1,2)
        plt.pcolormesh(x, y, np.abs(f_A_nosub[1]-f_A_nosub[0]))
        plt.xlabel(p2)
        plt.ylabel(p1)
        plt.title('difference A')

    
    
    plt.show()

