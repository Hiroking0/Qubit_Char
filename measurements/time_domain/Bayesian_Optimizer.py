# -*- coding: utf-8 -*-
"""
Script for parameter optimization

"""

import sys
sys.path.append("../../")
from lib import run_funcs
from lib import data_process as dp
from lib import wave_construction as be
from instruments.alazar import ATS9870_NPT as npt
from instruments.alazar import atsapi as ats
import yaml
import numpy as np

# ----------------- additional imports 
from lib.data_process import get_p1_p2, average_all_iterations

from hyperopt import fmin, tpe, hp


# if __name__ == "__main__":
    
    # f = open('./general_config.yaml','r')
    # params = yaml.safe_load(f)
    # f.close()

def run_system(params):
    with open('./general_config.yaml', 'r') as f:
        static_params = yaml.safe_load(f)
    print(params)
    #start_time is the time between triggering the AWG and start of the qubit pulse
    
    num_patterns = 2
    pattern_repeat = static_params['pattern_repeat']
    decimation = static_params['decimation']

    awg = be.get_awg()
    
    for i in range(num_patterns):
        awg.set_seq_element_loop_cnt(i+1, static_params['pattern_repeat'])

    board = ats.Board(systemId = 1, boardId = 1)
    npt.ConfigureBoard(board)
    
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    run_funcs.init_params(params)
    


    chA_avgs_sub, chB_avgs_sub, chA_avgs_nosub, chB_avgs_nosub, mag_sub, mag_nosub = run_funcs.run_and_acquire( awg,
                                                                                                                board,
                                                                                                                static_params,
                                                                                                                2,
                                                                                                                False,
                                                                                                                "./bay"
                                                                                                                )
    

    # def plot_all(chA, chB, num_patterns, pattern_repeat, seq_repeat, avg_start, avg_length, large_data_plot = False):
    return (np.average(mag_sub[0]), np.average(mag_sub[1]))
    

def objective(params):
    # Call the run_system function with the current parameter values
    output1, output2 = run_system(params)

    # Calculate the distance between the two outputs
    distance = np.abs(output1 - output2)

    # Return the negative distance (as Hyperopt minimizes the objective)
    return -distance



if __name__ == "__main__":
    
    with open('./general_config.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Define the search space for the optimization
    space = {
        'set_wr': hp.uniform('set_wr', 7.148, 7.1492),
        'set_wq': hp.uniform('set_wq', 3.31, 3.325),
        'set_pr': hp.uniform('set_pr', 12, 17),
        'set_pq': hp.uniform('set_pq', 12, 17),
        'set_p_twpa': hp.uniform('set_p_twpa', -20, -18),
        'set_w_twpa': hp.uniform('set_w_twpa', 7, 8.8),
        'set_r_att': hp.quniform('set_r_att', 40, 41, 1),
        'set_q_att': hp.quniform('set_q_att', 13, 21, 1)
        }

    best_params = fmin(objective, space, algo=tpe.suggest, max_evals=30)

    # # Update the params dictionary with the optimal parameter values
    # params.update(best_params)

    print('Optimal parameters:', best_params)

    # set_wr: 7.08370
    # set_wq: 2.93785 #2.93785  #2.9357
    # set_pr: 12.5
    # set_pq: 17 #or 16.4
    # set_att: 14






    