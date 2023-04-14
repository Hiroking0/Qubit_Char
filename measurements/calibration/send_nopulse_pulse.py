# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:38:30 2022

@author: lqc
"""
import sys
sys.path.append("../../")

#import npp_funcs as nppf
from lib import wave_construction as be
import yaml
from lib import run_funcs

#sends two patterns to AWG
#one pattern with no pulse, one with pulse
#readout on C2M1
#pattern on Ch1
#number of patterns = 2
def get_nopi_pi_group(
                start_time, #pulse start time
                q_duration, #pulse duration
                readout_start, #readout
                readout, #readout duration
                wait_time, #dead time after readout and end of pattern
                decimation):
    
    start_time = int(start_time/decimation)
    q_duration = int(q_duration/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    wait_time = int(wait_time/decimation)
    
    
    #start, duration, amplitude, channel, sweep_type, sweep_end, sweep_step, readout, wait_time
    p1 = be.Sweep_Pulse(start_time, q_duration, amplitude = 0, channel = 1, sweep_param = 'amplitude', sweep_stop = 1.1, sweep_step = 1)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1, wait_time = wait_time)
    pg = be.PulseGroup([p1, ro])
    return pg


if __name__ == "__main__":

    f = open('../general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()

    name = params['name']
    decimation = params['decimation']
    readout_start = params['readout_start']
    readout_dur = params['readout_duration']
    
    gap = params['nop_p_q_gap']
    q_duration = params['nop_p_q_dur']
    q_start = readout_start - gap - q_duration
    
    
    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    wait_time = zero_length * zero_multiple
    
    pattern_rep = params['pattern_repeat']
    
    num_patterns = 2
    
    awg = be.get_awg()
    
    
    readout_trigger_offset = params['readout_trigger_offset']
    be.READOUT_TRIGGER_OFFSET = readout_trigger_offset
    pg = get_nopi_pi_group(
                    start_time = q_start,
                    q_duration = q_duration,
                    readout_start = readout_start,
                    readout = readout_dur,
                    wait_time = wait_time,
                    decimation = decimation)

    pg.show(decimation, True)
    
    #pg.send_waves_awg(awg, name, pattern_rep, zero_length, zero_multiple, readout_trigger_offset, decimation)
    #run_funcs.initialize_awg(awg, num_patterns, pattern_rep, decimation)
    
    awg.close()
    
    
