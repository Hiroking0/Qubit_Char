# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:38:30 2022

@author: lqc
"""

import sys
sys.path.append("../../")
from lib import wave_construction as be
from lib import run_funcs
#import wave_construction as be
import yaml


def get_pulse_group(q_duration, #pulse duration
                first_pulse_start,
                second_pulse_start,
                first_pulse_final_start,
                first_pulse_step,
                readout_start, #readout
                readout, #readout duration
                wait_time,
                decimation = 1):
    q_duration = int(q_duration/decimation)
    first_pulse_start = int(first_pulse_start/decimation)
    second_pulse_start = int(second_pulse_start/decimation)
    first_pulse_final_start = int(first_pulse_final_start/decimation)
    first_pulse_step = int(first_pulse_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    wait_time = int(wait_time/decimation)
    
    
    be.READOUT_TRIGGER_OFFSET = readout_trigger_offset
    
    p1 = be.Sweep_Pulse(first_pulse_start, q_duration, amplitude = 1, channel = 1, 
                        sweep_param = 'start',
                        sweep_stop = first_pulse_final_start,
                        sweep_step = first_pulse_step)
    
    p2 = be.Pulse(second_pulse_start, q_duration, amplitude = 1, channel = 1)
    
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1, wait_time = wait_time)
    pg = be.PulseGroup([p1, p2, ro])
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
    
    q_duration = params['ramsey_q_dur']
    
    gap2 = params['ramsey_gap_2']
    g1_init = params['ramsey_gap_1_init']
    g1_final = params['ramsey_gap_1_final']
    first_pulse_step = params['time_domain_step']
    
    second_pulse_start = readout_start - gap2 - q_duration
    first_pulse_final_start = readout_start - gap2 - q_duration - g1_init - q_duration + first_pulse_step
    first_pulse_start = readout_start - gap2 - q_duration - g1_final - q_duration
    
    
    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    wait_time = zero_length * zero_multiple
    
    num_patterns = int((first_pulse_final_start - first_pulse_start)/first_pulse_step)
    
    print("Number of patterns: " + str(num_patterns))
    
    pattern_repeat = params['pattern_repeat']
    seq_repeat = params['seq_repeat']
    
    acq_multiples = params['acq_multiples']
    samples_per_ac = 256*acq_multiples #length of acquisition in nS must be n*256
    
    
    wave_len = readout_start + readout + wait_time
    awg = be.get_awg()
    
    readout_trigger_offset = params['readout_trigger_offset']
    #be.READOUT_TRIGGER_OFFSET = readout_trigger_offset
    
    pg = get_pulse_group(q_duration, #pulse duration
                    first_pulse_start,
                    second_pulse_start,
                    first_pulse_final_start,
                    first_pulse_step,
                    readout_start, #readout
                    readout, #readout duration
                    wait_time,
                    decimation)
    pg.show(decimation, True)
    
    #these two lines are not needed for displaying the pulse
    pg.send_waves_awg(awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset)
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, wave_len, decimation)
    
    awg.close()

