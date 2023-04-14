# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:25:11 2023

@author: lqc
"""

import send_rabi
import yaml


if __name__ == "__main__":
    
    f = open('../general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()

    name = params['name']
    decimation = params['decimation']
    #seq_nopi_pi(awg, name)
    #time.sleep(.1)
    #run_nopi_pi(awg, 10, .1)
    
    #start_time is the time between triggering the AWG and start of the qubit pulse
    #q_dur_stop is INclusive, changeable somewhere in wave construction probably
    
    start_time = params['rabi_pulse_start']
    q_dur_start = params['rabi_pulse_initial_duration']
    q_dur_stop = params['rabi_pulse_end_duration']
    
    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    wait_time = zero_length * zero_multiple
    
    num_patterns = params['num_patterns']
    
    q_dur_step = int((q_dur_stop - q_dur_start)/(num_patterns))
    
    #print(q_dur_step)
    readout_start = params['readout_start']
    readout = params['readout_duration']
    pattern_repeat = params['pattern_repeat']
    seq_repeat = params['seq_repeat']
    
    acq_multiples = params['acq_multiples']
    samples_per_ac = 256*acq_multiples #length of acquisition in nS must be n*256
    
    
    
    wave_len = readout_start + readout + wait_time

    pg = send_rabi.get_pulse_group(start_time, q_dur_start, q_dur_stop, q_dur_step, readout_start, readout, wait_time)
    pg.show(True)
    pg.to_file(name)
    
    readout_trigger_offset = params['readout_trigger_offset']
    
    #pg.send_waves_awg(awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset)
    
    #run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, wave_len)


