

import sys
sys.path.append("../")

#import npp_funcs as nppf
from lib import wave_construction as be
import yaml
from lib import run_funcs


def get_nopi_pi_group(
                start_time, #pulse start time
                q_duration, #pulse duration
                readout_start, #readout
                readout, #readout duration
                readout_trigger_offset,
                decimation):
    
    start_time = int(start_time/decimation)
    q_duration = int(q_duration/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    
    #start, duration, amplitude, channel, sweep_type, sweep_end, sweep_step, readout
    p1 = be.Sweep_Pulse(start_time, q_duration, amplitude = 0, channel = 1, sweep_param = 'amplitude', sweep_stop = 1.1, sweep_step = 1)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([p1, ro])
    return pg



def get_readout_group(
                readout_start, #readout
                readout, #readout duration
                readout_trigger_offset,
                decimation):
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    
    #start, duration, amplitude, channel, sweep_type, sweep_end, sweep_step, readout
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([ro])
    return pg




def get_T1_pulse_group(q_dur, #pulse start time
                q_start_start, #pulse duration
                q_start_end,
                q_start_step,
                readout_start, #readout
                readout, #readout 
                readout_trigger_offset,
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




def get_echo_pulse_group(pi_dur, #pulse start time
                gap_2,
                t_start, #pulse duration
                t_stop,
                t_step,
                readout_start, #readout
                readout, #readout duration
                readout_trigger_offset,
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



def get_echo1_pulse_group(pi_dur, #pulse start time
                gap_2,
                t_start, #pulse duration
                t_stop,
                t_step,
                readout_start, #readout
                readout, #readout duration
                readout_trigger_offset,
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
    
    y_pulse = be.Sweep_Pulse(p2_start, pi_dur, amplitude = 1, channel = 1, sweep_param = 'start', sweep_stop = p2_end, sweep_step = int(t_step/2))
    
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    
    pg = be.PulseGroup([x_pulse_1, y_pulse, x_pulse_2, ro])
    return pg






def get_rabi_pulse_group(start_time, #pulse start time
                q_dur_start, #pulse duration
                q_dur_stop,
                q_dur_step,
                readout_start, #readout
                readout, #readout duration
                readout_trigger_offset,
                decimation = 1):
    
    start_time = int(start_time/decimation)
    q_dur_start = int(q_dur_start/decimation)
    q_dur_stop = int(q_dur_stop/decimation)
    q_dur_step = int(q_dur_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    
    
    p1 = be.Sweep_Pulse(start_time, q_dur_start, amplitude = 1, channel = 1, sweep_param = 'duration', sweep_stop = q_dur_stop, sweep_step = q_dur_step)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([p1, ro])
    #pg.send_waves_awg(awg, "hi", 5)
    return pg


def get_ramsey_pulse_group(q_duration, #pulse duration
                first_pulse_start,
                second_pulse_start,
                first_pulse_final_start,
                first_pulse_step,
                readout_start, #readout
                readout, #readout duration
                readout_trigger_offset,
                decimation = 1):
    q_duration = int(q_duration/decimation)
    first_pulse_start = int(first_pulse_start/decimation)
    second_pulse_start = int(second_pulse_start/decimation)
    first_pulse_final_start = int(first_pulse_final_start/decimation)
    first_pulse_step = int(first_pulse_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    
    p1 = be.Sweep_Pulse(first_pulse_start, q_duration, amplitude = 1, channel = 1, 
                        sweep_param = 'start',
                        sweep_stop = first_pulse_final_start,
                        sweep_step = first_pulse_step)
    
    p2 = be.Pulse(second_pulse_start, q_duration, amplitude = 1, channel = 1)
    
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([p1, p2, ro])
    return pg

def get_pg(params):

    #First get params that are used by all measurements (readout values, name)
    #measurement will describe which pulse to send
    measurement = params['measurement']

    decimation = params['decimation']
    readout_start = params['readout_start']
    readout = params['readout_duration']

    readout_trigger_offset = params['readout_trigger_offset']
    step = params['time_domain_step']

    match measurement:
        case 'T1':
            q_duration = params['T1_q_dur']
            init_gap = params['T1_init_gap']
            final_gap = params['T1_final_gap']

            q_start_start = readout_start - final_gap - q_duration
            q_start_end = readout_start - init_gap - q_duration + step
            num_patterns = int((q_start_end - q_start_start)/step)
            pg = get_T1_pulse_group(q_duration, q_start_start, q_start_end, step, readout_start, readout, readout_trigger_offset, decimation)

        case 'rabi':
            q_dur_start = params['rabi_pulse_initial_duration']
            q_dur_stop = params['rabi_pulse_end_duration'] + step
            gap = params['rabi_pulse_gap']

            start_time = readout_start - gap - q_dur_start
            num_patterns = int((q_dur_stop - q_dur_start)/step)
            pg = get_rabi_pulse_group(start_time, q_dur_start, q_dur_stop, step, readout_start, readout, readout_trigger_offset, decimation)

        case 'ramsey':
            gap2 = params['ramsey_gap_2']
            g1_init = params['ramsey_gap_1_init']
            g1_final = params['ramsey_gap_1_final']
            
            q_duration = params['ramsey_q_dur']
            second_pulse_start = readout_start - gap2 - q_duration
            first_pulse_final_start = readout_start - gap2 - q_duration - g1_init - q_duration + step
            first_pulse_start = readout_start - gap2 - q_duration - g1_final - q_duration
            num_patterns = int((first_pulse_final_start - first_pulse_start)/step)
            pg = get_ramsey_pulse_group(q_duration, #pulse duration
                                        first_pulse_start,
                                        second_pulse_start,
                                        first_pulse_final_start,
                                        step,
                                        readout_start, #readout
                                        readout, #readout duration
                                        readout_trigger_offset,
                                        decimation)

        case 'npp':
            gap = params['nop_p_q_gap']
            q_duration = params['nop_p_q_dur']
            q_start = readout_start - gap - q_duration
            num_patterns = 2
            pg = get_nopi_pi_group(start_time = q_start,
                                    q_duration = q_duration,
                                    readout_start = readout_start,
                                    readout = readout,
                                    readout_trigger_offset = readout_trigger_offset,
                                    decimation = decimation)

        case 'readout':
            num_patterns = 1
            pg = get_readout_group(readout_start = readout_start,
                            readout = readout,
                            readout_trigger_offset = readout_trigger_offset,
                            decimation = decimation)

        case 'echo':
            gap_2 = params['echo_gap_2']
            t_initial = params['echo_initial_t']
            t_final = params['echo_final_t']
            pi_dur = params['echo_pi_pulse']
            num_patterns = int((t_final - t_initial)/step)
            pg = get_echo_pulse_group(pi_dur, gap_2, t_initial, t_final, step, readout_start, readout, readout_trigger_offset, decimation)

        case 'echo_1ax':
            gap_2 = params['echo_gap_2']
            t_initial = params['echo_initial_t']
            t_final = params['echo_final_t']
            pi_dur = params['echo_pi_pulse']
            num_patterns = int((t_final - t_initial)/step)
            pg = get_echo1_pulse_group(pi_dur, gap_2, t_initial, t_final, step, readout_start, readout, readout_trigger_offset, decimation)

    print(num_patterns)
    return pg

