# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:38:30 2022

@author: lqc
"""

import sys
sys.path.append("../../")

from lib import wave_construction as be
from lib import run_funcs
import yaml

def get_pulse_group(pi_dur, #pulse start time
                gap_2,
                t_start, #pulse duration
                t_stop,
                t_step,
                readout_start, #readout
                readout, #readout duration
                decimation = 1):
    pi_dur = int(pi_dur/decimation)
    gap_2 = int(gap_2/decimation)
    t_start = int(t_start/decimation)
    t_stop = int(t_stop/decimation)
    t_step = int(t_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    
    
    x_pulse_2 = be.Pulse(readout_start - gap_2 - int(pi_dur/2) , int(pi_dur/2), 1, 1)
    
    
    first_pulse_start = readout_start - gap_2 - (t_stop) - (2*pi_dur)
    first_pulse_end = readout_start - gap_2 - (t_start) - (2*pi_dur) + t_step
    
    x_pulse_1 = be.Sweep_Pulse(first_pulse_start, int(pi_dur/2), amplitude = 1, channel = 1, 
                        sweep_param = 'start',
                        sweep_stop = first_pulse_end,
                        sweep_step = t_step)
    
    
    p2_start = readout_start - gap_2 - int(t_stop/2) - int(3*pi_dur/2)
    p2_end = readout_start - gap_2 - int(t_start/2) - int(3*pi_dur/2) + int(t_step/2)
    #p2_end += half_t_step
    
    y_pulse = be.Sweep_Pulse(p2_start, pi_dur, amplitude = 1, channel = 2, sweep_param = 'start', sweep_stop = p2_end, sweep_step = int(t_step/2))
    
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    
    pg = be.PulseGroup([x_pulse_1, y_pulse, x_pulse_2, ro])
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
    
    
    t2_step = params['time_domain_step']
    gap_2 = params['echo_gap_2']
    t_initial = params['echo_initial_t']
    t_final = params['echo_final_t']
    pi_dur = params['echo_pi_pulse']

    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    
    num_patterns = int((t_final - t_initial)/t2_step)
    print(num_patterns)
    
    pattern_repeat = params['pattern_repeat']
    
    awg = be.get_awg()
    
    
    
    pg = get_pulse_group(pi_dur, gap_2, t_initial, t_final, t2_step, readout_start, readout, decimation)
    pg.show(decimation = decimation, subplots = False)
    
    readout_trigger_offset = int(params['readout_trigger_offset']/decimation)
    
    pg.send_waves_awg(awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset, decimation)
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    
    awg.close()

