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
from datetime import datetime

def int_eval(data):
    return eval(str(data))

def eval_yaml(yaml):
    string_params = ['name','measurement','shape','p1','p2']

    for key,val in yaml.items():
        if isinstance(val,dict) == False and key not in string_params :
            yaml[key] = int_eval(val)
        elif isinstance(val,dict) == True:
            for subkey, subval in val.items():
                if subkey not in string_params:
                    yaml[key][subkey] = int_eval(subval)
    return yaml


if __name__ == "__main__":
    
    f = open('./sbb_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    params = eval_yaml(params)
    #name = params['name']


    
    decimation = params['decimation']
    
    pattern_repeat = params['pattern_repeat']
    seq_repeat = params['seq_repeat']

    awg = be.get_awg()

    num_patterns = awg.get_seq_length()
    
    if params['name'] == 'auto':
        name = params['measurement'] + "_" + str(num_patterns)
    else:
        name = params['name']


    #w_len = awg.get_waveform_lengths(name + "_1_0")
    wait_time = params['zero_length'] * params['zero_multiple']# + w_len
    wait_time *= seq_repeat * pattern_repeat * num_patterns
    wait_time /= 1e9
    wait_time += .3
    print(f'estimated wait time is =~{wait_time} seconds')

    if params['measurement']=='effect_temp':
        run_funcs.turn_on_3rf()
    elif params['measurement']=='readout':
        run_funcs.turn_on_rf()
    else:
        run_funcs.turn_on_2rf()
    
    board = ats.Board(systemId = 1, boardId = 1)
    npt.ConfigureBoard(board)
    
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    run_funcs.init_params(params)
    raw_path = filedialog.askdirectory()
    now = datetime.now()
    date = now.strftime("%m%d_%H%M%S")
    path = raw_path + "/" + name + "_" + date


    data = run_funcs.run_and_acquire(awg,
                                    board,
                                    params,
                                    num_patterns,
                                    path)

    run_funcs.turn_off_inst()
    
    j_file = open(path+"_json.json", 'w')
    json.dump(params, j_file, indent = 4)
    j_file.close()
    data.save(path)
    
    if params['measurement'] == 'readout' or params['measurement'] == 'npp':
        time_step = 1
    else:
        time_step = params[params['measurement']]['step']

    
    dp.plot_np_file(data, time_step, path)
