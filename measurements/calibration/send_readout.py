# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:38:30 2022

@author: lqc
"""
import sys
sys.path.append("../../")

#import npp_funcs as nppf
from lib import wave_construction as be
from lib import run_funcs
import yaml


def get_readout_group(
                readout_start, #readout
                readout, #readout duration
                wait_time, #dead time after readout and end of pattern
                decimation):
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    wait_time = int(wait_time/decimation)
    
    #start, duration, amplitude, channel, sweep_type, sweep_end, sweep_step, readout, wait_time
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1, wait_time = wait_time)
    pg = be.PulseGroup([ro])
    return pg



if __name__ == "__main__":

    f = open('../general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()

    name = params['name']
    
    decimation = params['decimation']
    readout_start = params['readout_start']
    readout_dur = params['readout_duration']
    
    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    wait_time = zero_length * zero_multiple
    
    pattern_rep = params['pattern_repeat']
    seq_rep = params['seq_repeat']
    num_patterns = 1

    #start_time is the time between triggering the AWG and start of the qubit pulse
    

    #time after end of readout
    
    wave_len = readout_start + readout_dur + wait_time

    awg = be.get_awg()

    
    readout_trigger_offset = int(params['readout_trigger_offset']/decimation)
    
    pg = get_readout_group(
                    readout_start = readout_start,
                    readout = readout_dur,
                    wait_time = wait_time,
                    decimation = decimation)
    pg.show(decimation, True)
    
    pg.send_waves_awg(awg, name, pattern_rep, zero_length, zero_multiple, readout_trigger_offset, decimation)
    run_funcs.initialize_awg(awg, num_patterns, pattern_rep, wave_len, decimation)
    
    awg.close()    

