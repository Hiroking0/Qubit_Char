# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:38:30 2022

@author: lqc
"""

import sys
sys.path.append("../")
from lib import run_funcs
from lib import data_process as dp
from lib import wave_construction as be
from instruments.alazar import ATS9870_NPT as npt
from instruments.alazar import atsapi as ats
import yaml
import tkinter.filedialog as filedialog
import os
import json
#import numpy as np

if __name__ == "__main__":
    
    f = open('./general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    name = params['name']
    decimation = params['decimation']
    
    
    #start_time is the time between triggering the AWG and start of the qubit pulse
    #zero_length = params['zero_length']
    #zero_multiple = params['zero_multiple']
    #wait_time = zero_length * zero_multiple
    
    
    readout_start = params['readout_start']
    readout_dur = params['readout_duration']
    pattern_repeat = params['pattern_repeat']
    seq_repeat = params['seq_repeat']
    acq_multiples = params['acq_multiples']
    samples_per_ac = 256*acq_multiples #length of acquisition in nS must be n*256

    avg_start = params['avg_start']
    avg_length = params['avg_length']

    #wave_len = readout_start + readout_dur + wait_time

    time_step = params['time_domain_step']

    awg = be.get_awg()
    
    num_patterns = awg.get_seq_length()
    
    path = filedialog.askdirectory() + "/" + name + "_"
    #os.mkdir(path)
    #path = path + "/" + name + "_"

    
    for i in range(num_patterns):
        awg.set_seq_element_loop_cnt(i+1, params['pattern_repeat'])

    board = ats.Board(systemId = 1, boardId = 1)
    npt.ConfigureBoard(board)
    
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    run_funcs.init_params(params)
    
    #saves raw data if only readout or readout + pulse
    if num_patterns < 3 and seq_repeat * pattern_repeat <= 10000:
        save_raw = True
    else:
        save_raw = False
    
    chA, chB = run_funcs.run_and_acquire(
                awg,
                board,
                params,
                num_patterns,
                save_raw,
                path
                )
    
    if num_patterns < 3 and seq_repeat * pattern_repeat < 10000:
        dp.plot_all(chA, chB, num_patterns, pattern_repeat, seq_repeat, params['avg_start'], params['avg_length'], large_data_plot = False)
        #dp.plot_np_file(num_patterns, pattern_repeat, seq_repeat, time_step, path)
    else:
        dp.plot_np_file(num_patterns, pattern_repeat, seq_repeat, time_step, path)
    #dp.plot_np_file(num_patterns, pattern_repeat, seq_repeat, time_step, path)
    
    