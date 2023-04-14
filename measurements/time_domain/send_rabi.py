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


def get_pulse_group(start_time, #pulse start time
                q_dur_start, #pulse duration
                q_dur_stop,
                q_dur_step,
                readout_start, #readout
                readout, #readout duration
                wait_time,
                decimation = 1):
    
    start_time = int(start_time/decimation)
    q_dur_start = int(q_dur_start/decimation)
    q_dur_stop = int(q_dur_stop/decimation)
    q_dur_step = int(q_dur_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    wait_time = int(wait_time/decimation)
    
    
    be.READOUT_TRIGGER_OFFSET = readout_trigger_offset
    p1 = be.Sweep_Pulse(start_time, q_dur_start, amplitude = 1, channel = 1, sweep_param = 'duration', sweep_stop = q_dur_stop, sweep_step = q_dur_step)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1, wait_time = wait_time)
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
    pattern_repeat = params['pattern_repeat']
    seq_repeat = params['seq_repeat']
    
    #start_time is the time between triggering the AWG and start of the qubit pulse
    #q_dur_stop is INclusive, changeable somewhere in wave construction probably
    
    #start_time = params['rabi_pulse_start']
    q_dur_start = params['rabi_pulse_initial_duration']
    q_dur_step = int(params['time_domain_step'])
    q_dur_stop = params['rabi_pulse_end_duration'] + q_dur_step
    gap = params['rabi_pulse_gap']
    start_time = readout_start - gap - q_dur_start
    
    
    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    wait_time = zero_length * zero_multiple
    
    #num_patterns = params['num_patterns']
    
    #q_dur_step = int((q_dur_stop - q_dur_start)/(num_patterns))
    num_patterns = int((q_dur_stop - q_dur_start)/q_dur_step)
    
    print("Number of patterns: " + str(num_patterns))
    
    
    acq_multiples = params['acq_multiples']
    samples_per_ac = 256*acq_multiples #length of acquisition in nS must be n*256
    
    
    
    wave_len = readout_start + readout + wait_time
    awg = be.get_awg()
    
    readout_trigger_offset = params['readout_trigger_offset']
    #be.READOUT_TRIGGER_OFFSET = readout_trigger_offset
    
    pg = get_pulse_group(start_time, q_dur_start, q_dur_stop, q_dur_step, readout_start, readout, wait_time, decimation)
    pg.show(decimation, True)
    
    #these two lines are not needed for displaying the pulse
    pg.send_waves_awg(awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset)
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    
    awg.close()

