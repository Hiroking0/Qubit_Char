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
import json
import numpy as np

if __name__ == "__main__":
    
    f = open('./general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    name = params['name']
    decimation = params['decimation']
    
    pattern_repeat = params['pattern_repeat']
    seq_repeat = params['seq_repeat']
    
    readout_dur = params[params['measurement']]['readout_duration']
    readout_trigger_offset = params['readout_trigger_offset']
    acq_multiples = int((readout_dur + readout_trigger_offset)/256) + 10
    samples_per_ac = 256*acq_multiples #length of acquisition in nS must be n*256

    avg_start = params['avg_start']
    avg_length = params['avg_length']

    awg = be.get_awg()
    num_patterns = awg.get_seq_length()
    
    path = filedialog.askdirectory() + "/" + name + "_"
    
    for i in range(num_patterns):
        awg.set_seq_element_loop_cnt(i+1, params['pattern_repeat'])

    board = ats.Board(systemId = 1, boardId = 1)
    npt.ConfigureBoard(board)
    
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    run_funcs.init_params(params)
    
    #saves raw data if only readout or readout + pulse
    if num_patterns < 3 and seq_repeat * pattern_repeat * num_patterns <= 20000:
        save_raw = True
    else:
        save_raw = False
    
    chA_avgs_sub, chB_avgs_sub, chA_avgs_nosub, chB_avgs_nosub, mag_sub, mag_nosub = run_funcs.run_and_acquire(
                awg,
                board,
                params,
                num_patterns,
                save_raw,
                path
                )
    j_file = open(path+"json.json", 'w')
    json.dump(params, j_file, indent = 4)
    j_file.close()
    
    np.save(path + "chA_sub", chA_avgs_sub)
    np.save(path + "chB_sub", chB_avgs_sub)
    np.save(path + "chA_nosub", chA_avgs_nosub)
    np.save(path + "chB_nosub", chB_avgs_nosub)
    np.save(path + "mag_sub", mag_sub)
    np.save(path + "mag_nosub", mag_nosub)
    
    
    if save_raw:
        (chA, chB) = dp.frombin(tot_samples = samples_per_ac, numAcquisitions = num_patterns*pattern_repeat*seq_repeat, channels = 2, name = path + "rawdata.bin")
        dp.plot_all(chA, chB, num_patterns, pattern_repeat, seq_repeat, params['avg_start'], params['avg_length'], large_data_plot = False)
        #dp.plot_np_file(num_patterns, pattern_repeat, seq_repeat, time_step, path)
    else:
        time_step = params[params['measurement']]['step']
        dp.plot_np_file(num_patterns, pattern_repeat, seq_repeat, time_step, path)
        
    #dp.plot_np_file(num_patterns, pattern_repeat, seq_repeat, time_step, path)
    
    