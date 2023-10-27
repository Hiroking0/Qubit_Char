# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:38:30 2022

@author: lqc
"""

import sys
sys.path.append("../../")
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
from time_domain.run_sequence import eval_yaml
from datetime import datetime

def int_eval(data):
    return eval(str(data))

def plot_subax(ax, x, y, title):
    for i in range(len(y)):
        ax.plot(x, y[i])
    ax.set_title(title)


if __name__ == "__main__":
    
    f = open('general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    yaml = eval_yaml(params)
    

    #name = params['name']

    directory = tkf.askdirectory()
    awg = be.get_awg()
    num_patterns = awg.get_seq_length()

    if params['name'] == 'auto':
        name = directory + "/" + params['measurement'] + "_" + str(num_patterns)
    else:
        name = directory + "/" + params['name']

    
    decimation = params['decimation']

    now = datetime.now()
    date = now.strftime("%m%d_%H%M%S")
    name = f"{name}_{date}"

    
    j_file = open(name+"_json.json", 'w')
    json.dump(params, j_file, indent = 4)
    j_file.close()

    zero_length = int_eval(params['zero_length'])
    zero_multiple = int_eval(params['zero_multiple'])
    wait_time = zero_length * zero_multiple
    
    pattern_repeat = int_eval(params['pattern_repeat'])
    seq_repeat = int_eval(params['seq_repeat'])
    
    p1start = int_eval(params['p1start'])
    p1stop = int_eval(params['p1stop'])
    p1step = int_eval(params['p1step'])
    
    p1 = params['p1']

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


    cAp_sub, cBp_sub, cAp_nosub, cBp_nosub, mags_sub, mags_nosub = run_funcs.single_sweep(name,
                                                                                            awg,
                                                                                            board,
                                                                                            num_patterns,
                                                                                            params,
                                                                                            live_plot = False)

    run_funcs.turn_off_inst()

    if params['measurement'] != 'readout' and params['p1'] == 'wq':
        ssb = params[params['measurement']]['ssb_freq']
        x = np.arange(p1start + ssb, p1stop + ssb, p1step)
    else:
        x = np.arange(p1start, p1stop, p1step)
    
    #plt.figure()

    fig, ax_array = plt.subplots(2,3)
    plot_subax(ax_array[0,0], x, cAp_sub, 'channel a sub')
    plot_subax(ax_array[0,1], x, cBp_sub, 'channel b sub')
    plot_subax(ax_array[0,2], x, mags_sub, 'mags sub')
    plot_subax(ax_array[1,0], x, cAp_nosub, 'channel a nosub')
    plot_subax(ax_array[1,1], x, cBp_nosub, 'channel b nosub')
    plot_subax(ax_array[1,2], x, mags_nosub, 'mags nosub')
    plt.savefig(name + "pic", dpi= 300, pad_inches = 0, bbox_inches = 'tight')
    #plt.figure()
    #diff = cBp_nosub[0] - cBp_nosub[1]
    #plt.plot(x, diff)
    
    plt.show()
    