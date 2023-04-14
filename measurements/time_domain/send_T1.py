# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:38:30 2022

@author: lqc
"""

import sys
sys.path.append("../../")
#import T1_funcs as T1

from lib import wave_construction as be
from lib import run_funcs
import yaml

def get_pulse_group(q_dur, #pulse start time
                q_start_start, #pulse duration
                q_start_end,
                q_start_step,
                readout_start, #readout
                readout, #readout duration
                decimation = 1):
    
    q_dur = int(q_dur/decimation)
    q_start_start = int(q_start_start/decimation)
    q_start_end = int(q_start_end/decimation)
    q_start_step = int(q_start_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    
    
    p1 = be.Sweep_Pulse(q_start_start, q_dur, amplitude = 1, channel = 1, sweep_param = 'start', sweep_stop = q_start_end, sweep_step = q_start_step)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([p1, ro])
    #pg.send_waves_awg(awg, "hi", 5)
    return pg


if __name__ == "__main__":
    
    f = open('../general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()

    name = params['name']
    decimation = params['decimation']
    readout_start = params['readout_start']
    readout = params['readout_duration']
                     
    
    #start_time is the time between triggering the AWG and start of the qubit pulse
    #q_dur_stop is INclusive, changeable somewhere in wave construction probably    
    
    q_duration = params['T1_q_dur']
    q_start_step = params['time_domain_step']
    init_gap = params['T1_init_gap']
    final_gap = params['T1_final_gap']
    
    q_start_start = readout_start - final_gap - q_duration
    q_start_end = readout_start - init_gap - q_duration + q_start_step
    

    print("start", q_start_start, "end", q_start_end)

    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    
    num_patterns = int((q_start_end - q_start_start)/q_start_step)
    print(num_patterns)
    
    
    pattern_repeat = params['pattern_repeat']
    
    awg = be.get_awg()
    
    
    
    pg = get_pulse_group(q_duration, q_start_start, q_start_end, q_start_step, readout_start, readout, decimation)
    pg.show(decimation, True)
    
    readout_trigger_offset = params['readout_trigger_offset']
    
    pg.send_waves_awg(awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset, decimation)
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    
    awg.close()

