

import sys
sys.path.append("../")

#import npp_funcs as nppf
from lib import wave_construction as be
import yaml
from lib import run_funcs
import math
import numpy as np

def int_eval(data):
    return eval(str(data))
def get_phase_pulse_group(
                            start_time,
                            q_duration,
                            readout_start,
                            readout,
                            frequency,
                            ssb_phase_start,
                            ssb_phase_end,
                            ssb_phase_step,
                            #ro_freq,
                            decimation,
                            shape):

    start_time = int(start_time/decimation)
    q_duration = int(q_duration/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    frequency *= decimation
    
    pulse_class = getattr(be, f'Phase_Sweep_{shape}')
    
    p1 = pulse_class(start_time,
                            q_duration,
                            amplitude = 1,
                            frequency = frequency,
                            phase = 0,
                            channel = 1,
                            sweep_stop = ssb_phase_end,
                            sweep_step = ssb_phase_step,
                            phase_adjust=False)
    p2 = pulse_class(start_time,
                            q_duration,
                            amplitude = 1,
                            frequency = frequency,
                            phase = np.radians(ssb_phase_start),
                            channel = 2,
                            sweep_stop = ssb_phase_end,
                            sweep_step = ssb_phase_step)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([p1, p2, ro])
    return pg

def get_sbb_freq_group( 
                        start_time,
                        q_duration,
                        readout_start,
                        readout,
                        ssb_freq_start,
                        ssb_freq_end,
                        ssb_freq_step,
                        phase,
                        #ro_freq,
                        decimation,
                        shape):
    
    start_time = int(start_time/decimation)
    q_duration = int(q_duration/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    ssb_freq_start *= decimation
    ssb_freq_end *= decimation
    ssb_freq_step *= decimation
    
    pulse_class = getattr(be, f'Freq_Sweep_{shape}')
    
    p1 = pulse_class(start_time,
                            q_duration,
                            amplitude = 1,
                            frequency = ssb_freq_start,
                            phase = 0,
                            channel = 1,
                            sweep_stop = ssb_freq_end,
                            sweep_step = ssb_freq_step)
    p2 = pulse_class(start_time,
                            q_duration,
                            amplitude = 1,
                            frequency = ssb_freq_start,
                            phase = np.radians(phase),
                            channel = 2,
                            sweep_stop = ssb_freq_end,
                            sweep_step = ssb_freq_step)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([p1, p2, ro])
    return pg

def get_single_pulse_group(
                start_time, #pulse start time
                q_duration, #pulse duration
                readout_start, #readout
                readout, #readout duration
                frequency,
                phase,
                #ro_freq,
                decimation,
                shape):
    
    start_time = int(start_time/decimation)
    q_duration = int(q_duration/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    frequency *= decimation
    
    pulse_class = getattr(be, f'Amp_Sweep_{shape}')
    
    p1 = pulse_class(start_time,
                            q_duration,
                            amplitude = 1,
                            frequency = frequency,
                            phase = 0,
                            channel = 1,
                            sweep_stop = 2,
                            sweep_step = 1)
    p2 = pulse_class(start_time,
                            q_duration,
                            amplitude = 1,
                            frequency = frequency,
                            phase = np.radians(phase),
                            channel = 2,
                            sweep_stop = 2,
                            sweep_step = 1)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([p1, p2, ro])
    return pg

def get_nopi_pi_group(
                start_time, #pulse start time
                q_duration, #pulse duration
                readout_start, #readout
                readout, #readout duration
                frequency,
                phase,
                #ro_freq,
                decimation,
                shape):
    
    start_time = int(start_time/decimation)
    q_duration = int(q_duration/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    frequency *= decimation
    
    pulse_class = getattr(be, f'Amp_Sweep_{shape}')
    
    p1 = pulse_class(start_time,
                            q_duration,
                            amplitude = 0,
                            frequency = frequency,
                            phase = 0,
                            channel = 1,
                            sweep_stop = 2,
                            sweep_step = 1)
    p2 = pulse_class(start_time,
                            q_duration,
                            amplitude = 0,
                            frequency = frequency,
                            phase = np.radians(phase),
                            channel = 2,
                            sweep_stop = 2,
                            sweep_step = 1)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([p1, p2, ro])
    return pg

def get_readout_group(
                readout_start, #readout
                readout, #readout duration
                decimation):
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    
    #start, duration, amplitude, channel, sweep_type, sweep_end, sweep_step, readout
    #ro = be.Sin_Readout(readout_start, readout, 1, freq)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([ro])
    return pg


def get_T1_pulse_group(q_dur, #pulse start time
                q_start_start, #pulse duration
                q_start_end,
                q_start_step,
                readout_start, #readout
                readout, #readout
                frequency,
                phase,
                decimation,
                shape):

    
    q_dur = int(q_dur/decimation)
    q_start_start = int(q_start_start/decimation)
    q_start_end = int(q_start_end/decimation)
    q_start_step = int(q_start_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    frequency *= decimation
    
    pulse_class = getattr(be, f'Start_Sweep_{shape}')
    p1 = pulse_class(q_start_start,
                              q_dur,
                              amplitude = 1,
                              frequency = frequency,
                              phase = 0,
                              channel = 1,
                              sweep_stop = q_start_end,
                              sweep_step = q_start_step)
    p2 = pulse_class(q_start_start,
                              q_dur,
                              amplitude = 1,
                              frequency = frequency,
                              phase = np.radians(phase),
                              channel = 2,
                              sweep_stop = q_start_end,
                              sweep_step = q_start_step)
    
    
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([p1, p2, ro])
    #pg.send_waves_awg(awg, "hi", 5)
    return pg




def get_echo_pulse_group(pi_dur, 
                gap_2,
                t_start,
                t_stop,
                t_step,
                readout_start, #readout
                readout, #readout duration
                decimation,
                shape):
    pi_dur = int(pi_dur/decimation)
    gap_2 = int(gap_2/decimation)
    t_start = int(t_start/decimation)
    t_stop = int(t_stop/decimation)
    t_step = int(t_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    #frequency *= decimation
    
    
    pulse_class = getattr(be, f'Sin_{shape}')
    #self, start, duration, amplitude, frequency, phase, channel, numsig=6)
    x_pulse_2 = pulse_class(readout_start - gap_2 - int(pi_dur/2) , int(pi_dur/2), 1, 0, 0, 1)
    
    
    first_pulse_start = readout_start - gap_2 - (t_stop) - (2*pi_dur)# - t_step
    first_pulse_end = readout_start - gap_2 - (t_start) - (2*pi_dur) + t_step
    
    
    
    start_sweep_class = getattr(be, f'Start_Sweep_{shape}')
    
    x_pulse_1 = start_sweep_class(first_pulse_start,
                                     int(pi_dur/2),
                                     amplitude = 1,
                                     frequency = 0,
                                     phase = 0,
                                     channel = 1,
                                     sweep_stop = first_pulse_end,
                                     sweep_step = t_step)
    
    

    
    p2_start = readout_start - gap_2 - int(t_stop/2) - int(3*pi_dur/2) #- int(t_step/2)
    p2_end = readout_start - gap_2 - int(t_start/2) - int(3*pi_dur/2) + int(t_step/2)
    #p2_end += half_t_step
    
    
    y_pulse = start_sweep_class(p2_start,
                                     pi_dur,
                                     amplitude = 1,
                                     frequency = 0,
                                     phase = 0,
                                     channel = 2,
                                     sweep_stop = p2_end,
                                     sweep_step = int(t_step/2))

    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    
    
    
    
    pg = be.PulseGroup([x_pulse_1, y_pulse, x_pulse_2, ro])
    return pg


def get_echo1_pulse_group(pi_dur, 
                gap_2,
                t_start,
                t_stop,
                t_step,
                readout_start, #readout
                readout, #readout duration
                decimation,
                shape):
    pi_dur = int(pi_dur/decimation)
    gap_2 = int(gap_2/decimation)
    t_start = int(t_start/decimation)
    t_stop = int(t_stop/decimation)
    t_step = int(t_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    
    
    pulse_class = getattr(be, f'Sin_{shape}')
    start_sweep_class = getattr(be, f'Start_Sweep_{shape}')
    #self, start, duration, amplitude, frequency, phase, channel, numsig=6)
    x_pulse_2 = pulse_class(readout_start - gap_2 - int(pi_dur/2) , int(pi_dur/2), 1, 0, 0, 1)
    
    
    first_pulse_start = readout_start - gap_2 - (t_stop) - (2*pi_dur)# - t_step
    first_pulse_end = readout_start - gap_2 - (t_start) - (2*pi_dur) + t_step
    
    x_pulse_1 = start_sweep_class(first_pulse_start,
                                     int(pi_dur/2),
                                     amplitude = 1,
                                     frequency = 0,
                                     phase = 0,
                                     channel = 1,
                                     sweep_stop = first_pulse_end,
                                     sweep_step = t_step)
    

    p2_start = readout_start - gap_2 - int(t_stop/2) - int(3*pi_dur/2) #- int(t_step/2)
    p2_end = readout_start - gap_2 - int(t_start/2) - int(3*pi_dur/2) + int(t_step/2)
    #p2_end += half_t_step
    
    y_pulse = start_sweep_class(p2_start,
                                     pi_dur,
                                     amplitude = 1,
                                     frequency = 0,
                                     phase = 0,
                                     channel = 1,
                                     sweep_stop = p2_end,
                                     sweep_step = int(t_step/2))
    

    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    
    pg = be.PulseGroup([x_pulse_1, y_pulse, x_pulse_2, ro])
    return pg


def get_rabi_pulse_group(start_time, #pulse start time
                q_dur_start, #pulse duration
                q_dur_stop,
                q_dur_step,
                readout_start, #readout
                readout, #readout duration
                frequency,
                phase,
                decimation,
                shape):
    
    start_time = int(start_time/decimation)
    q_dur_start = int(q_dur_start/decimation)
    q_dur_stop = int(q_dur_stop/decimation)
    q_dur_step = int(q_dur_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    frequency *= decimation
    
    
    pulse_class = getattr(be, f'Duration_Sweep_{shape}')
    
    #_(self, start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step):
    p1 = pulse_class(start_time,
                                 q_dur_start,
                                 amplitude = 1,
                                 frequency = frequency,
                                 phase = 0,
                                 channel = 1,
                                 sweep_stop = q_dur_stop,
                                 sweep_step = q_dur_step)
    p2 = pulse_class(start_time,
                                 q_dur_start,
                                 amplitude =1,
                                 frequency = frequency,
                                 phase = np.radians(phase),
                                 channel = 2,
                                 sweep_stop = q_dur_stop,
                                 sweep_step = q_dur_step)
    
    #p1 = be.Sweep_Pulse(start_time, q_dur_start, amplitude = 1, frequency=frequency, channel = 1, sweep_param = 'duration', sweep_stop = q_dur_stop, sweep_step = q_dur_step)
    
    
    
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    
    pg = be.PulseGroup([p1, p2, ro])
    #pg.send_waves_awg(awg, "hi", 5)
    return pg


def get_ramsey_pulse_group(q_duration, #pulse duration
                first_pulse_start,
                second_pulse_start,
                first_pulse_final_start,
                first_pulse_step,
                readout_start, #readout
                readout, #readout duration
                frequency,
                phase,
                decimation,
                shape):
    q_duration = int(q_duration/decimation)
    first_pulse_start = int(first_pulse_start/decimation)
    second_pulse_start = int(second_pulse_start/decimation)
    first_pulse_final_start = int(first_pulse_final_start/decimation)
    first_pulse_step = int(first_pulse_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    frequency *= decimation
    
    
    sweep_class = getattr(be, f'Start_Sweep_{shape}')
    single_class = getattr(be, f'Sin_{shape}')
    
    #self, start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step)
    p1c1 = sweep_class(first_pulse_start,
                        q_duration,
                        1,
                        frequency,
                        0,
                        1,
                        first_pulse_final_start,
                        first_pulse_step)
    
    
    p1c2 = sweep_class(first_pulse_start,
                        q_duration,
                        1,
                        frequency,
                        np.radians(phase),
                        2,
                        first_pulse_final_start,
                        first_pulse_step)
    
    #self, start, duration, amplitude, frequency, phase, channel
    p2c1 = single_class(second_pulse_start, q_duration, 1, frequency, 0, 1)
    p2c2 = single_class(second_pulse_start, q_duration, 1, frequency, np.radians(phase), 2)
    
    
    #p3 = be.Sin_Pulse(second_pulse_start, q_duration, amplitude = 1, frequency=frequency, channel = 1)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    
    
    
    pg = be.PulseGroup([p1c1, p1c2, p2c1, p2c2, ro])
    return pg

def get_amp_pg(q_duration,
               q_gap,
               amp_start,
               amp_stop,
               amp_step,
               readout_start,
               readout,
               frequency,
               phase,
               decimation,
               shape):
    
    
    sweep_class = getattr(be, f'Amp_Sweep_{shape}')
    
    #get_amp_pg(q_duration, q_gap, a_start, a_stop, step, readout_start, readout, wq_offset, phase, readout)
    
    q_duration = int(q_duration/decimation)
    q_gap = int(q_gap/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    
    q_start = readout_start - q_gap - q_duration
    
    
    #self, start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step
    p1 = sweep_class(q_start, q_duration, amp_start, frequency, phase = 0, channel = 1, sweep_stop = amp_stop, sweep_step = amp_step)
    p2 = sweep_class(q_start, q_duration, amp_start, frequency, phase = np.radians(phase), channel = 2, sweep_stop = amp_stop, sweep_step = amp_step)
    
    
    
    ro = be.Readout_Pulse(readout_start, readout, 1)
    pg = be.PulseGroup([p1, p2, ro])
    return pg



def get_et_pulse_group(ge_first_duration,
                       ge_second_duration,
                       gap_1,
                       gap_2,
                       rabi_start,
                       rabi_stop,
                       rabi_step,
                       readout_start,
                       readout_dur,
                       frequency_ge,
                       phase_ge,
                       freq_ef,
                       phase_ef,
                       decimation,
                       shape):
    
    
    start_sweep_class = getattr(be, f'Start_Sweep_{shape}')
    duration_sweep_class = getattr(be, f'Duration_Sweep_{shape}')
    single_class = getattr(be, f'Sin_{shape}')
    
    
    p1_start_init = readout_start - gap_2 - rabi_start - gap_1 - ge_first_duration - gap_1 - ge_second_duration
    p1_start_final = readout_start - gap_2 - rabi_stop - gap_1 - ge_first_duration  - gap_1 - ge_second_duration
    
    p1_start_init += rabi_step
    p1_start_final -= rabi_step
    #p1 has to be shifted by one step because duration sweeps from left to right
    #and start time sweeps from right to left.
    #This may be nice to fix in wave construction of sweep pulse.
    
    
    #self, start, duration, amplitude, frequency, sweep_param, sweep_stop, sweep_step, channel = None):
    
    pulse1 = start_sweep_class(p1_start_final,
                                  ge_first_duration,
                                  amplitude = 1,
                                  frequency = frequency_ge,
                                  phase = 0,
                                  channel = 1,
                                  sweep_stop = p1_start_init,
                                  sweep_step = rabi_step)
    
    p1c2 = start_sweep_class(p1_start_final,
                                  ge_first_duration,
                                  amplitude = 1,
                                  frequency = frequency_ge,
                                  phase = np.radians(phase_ge),
                                  channel = 2,
                                  sweep_stop = p1_start_init,
                                  sweep_step = rabi_step)
    
    p2_start_init = readout_start - gap_2 - rabi_start - gap_1 - ge_second_duration

    pulse2 = duration_sweep_class(p2_start_init,
                                     rabi_start,
                                     amplitude = 1,
                                     frequency = freq_ef,
                                     phase = 0,
                                     channel = 3,
                                     sweep_stop = rabi_stop,
                                     sweep_step = rabi_step)
    
    p2c4 = duration_sweep_class(p2_start_init,
                                     rabi_start,
                                     amplitude = 1,
                                     frequency = freq_ef,
                                     phase = np.radians(phase_ef),
                                     channel = 4,
                                     sweep_stop = rabi_stop,
                                     sweep_step = rabi_step)
    
    #self, start: int, duration: int, amplitude: float, frequency: float, channel: int
    p3_start = readout_start - gap_2 - ge_second_duration
    
    pulse3 = single_class(p3_start, ge_second_duration, amplitude = 1, frequency = frequency_ge, phase = 0, channel = 1)
    p3c2 = single_class(p3_start, ge_second_duration, amplitude = 1, frequency = frequency_ge, phase = np.radians(phase_ge), channel = 2)
    
    
    ro = be.Readout_Pulse(readout_start, readout_dur, 1)
    pg = be.PulseGroup([pulse1, p1c2, pulse2, p2c4, pulse3, p3c2, ro])
    
    return pg

def get_pg(params):

    #First get params that are used by all measurements (readout values, name)
    #measurement will describe which pulse to send
    measurement = params['measurement']
    readout_buffer = int(int_eval(params['readout_trigger_offset']) * 1.5)
    #readout_buffer = 0
    decimation = int_eval(params['decimation'])
    #readout_start = params['readout_start']
    
    readout_trigger_offset = int_eval(params['readout_trigger_offset'])
    
    params = params[measurement] if (measurement != 'echo_1ax') else params['echo']
    shape = params['shape']
    readout = int_eval(params['readout_duration'])
    if measurement != 'readout' and measurement != 'effect_temp':
        if measurement != 'sbb_freq_sweep':
            wq_offset = int_eval(params['ssb_freq'])
        if measurement != 'sbb_phase_sweep':
            phase = int_eval(params['ssb_phase'])
        

    match measurement:
        case 'T1':
            q_duration = int_eval(params['T1_q_dur'])
            init_gap = int_eval(params['T1_init_gap'])
            final_gap = int_eval(params['T1_final_gap'])
            step = int_eval(params['step'])
            readout_start = q_duration + final_gap + readout_buffer
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            q_start_start = readout_start - final_gap - q_duration
            q_start_end = readout_start - init_gap - q_duration + step
            num_patterns = int((q_start_end - q_start_start)/step)
            pg = get_T1_pulse_group(q_duration, q_start_start, q_start_end, step, readout_start, readout, wq_offset, phase, decimation, shape)

        case 'rabi':
            step = int_eval(params['step'])
            q_dur_start = int_eval(params['rabi_pulse_initial_duration'])
            q_dur_stop = int_eval(params['rabi_pulse_end_duration']) + step
            gap = int_eval(params['rabi_pulse_gap'])
            
            readout_start = gap + q_dur_stop + readout_buffer
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            start_time = readout_start - gap - q_dur_start
            num_patterns = int((q_dur_stop - q_dur_start)/step)
            pg = get_rabi_pulse_group(start_time, q_dur_start, q_dur_stop, step, readout_start, readout, wq_offset, phase, decimation, shape)

        case 'ramsey':
            gap2 = int_eval(params['ramsey_gap_2'])
            g1_init = int_eval(params['ramsey_gap_1_init'])
            g1_final = int_eval(params['ramsey_gap_1_final'])
            step = int_eval(params['step'])
            q_duration = int_eval(params['ramsey_q_dur']) # pi/2 pulse
            
            
            
            readout_start = readout_buffer + gap2 + 2*q_duration + g1_final
            #Round to nearest multiple of decimation 
            readout_start = decimation * math.ceil(readout_start/decimation)
            
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
                                        wq_offset,
                                        phase,
                                        decimation,
                                        shape)

        case 'npp':
            gap = int_eval(params['nop_p_q_gap'])
            q_duration = int_eval(params['nop_p_q_dur'])
           
            readout_start = gap + q_duration + readout_buffer
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            q_start = readout_start - gap - q_duration
            num_patterns = 2
            pg = get_nopi_pi_group(start_time = q_start,
                                    q_duration = q_duration,
                                    readout_start = readout_start,
                                    readout = readout,
                                    frequency = wq_offset,
                                    phase = phase,
                                    #ro_freq = wr_offset,
                                    decimation = decimation,
                                    shape = shape)

        case 'readout':
            num_patterns = 1
            readout_start = readout_buffer
            readout_start = decimation * math.ceil(readout_start/decimation)
            pg = get_readout_group(readout_start = readout_start,
                            readout = readout,
                            decimation = decimation)
            

        case 'echo':
            gap_2 = int_eval(params['echo_gap_2'])
            t_initial = int_eval(params['echo_initial_t'])
            t_final = int_eval(params['echo_final_t'])
            pi_dur = int_eval(params['echo_pi_pulse'])
            step = int_eval(params['step'])
            
            readout_start = readout_buffer + 2*pi_dur + t_final + gap_2
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            num_patterns = int((t_final - t_initial)/step)
            pg = get_echo_pulse_group(pi_dur, gap_2, t_initial, t_final, step, readout_start, readout, decimation, shape)

        case 'echo_1ax':
            gap_2 = int_eval(params['echo_gap_2'])
            t_initial = int_eval(params['echo_initial_t'])
            t_final = int_eval(params['echo_final_t'])
            pi_dur = int_eval(params['echo_pi_pulse'])
            step = int_eval(params['step'])
            readout_start = readout_buffer + 2*pi_dur + t_final + gap_2
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            num_patterns = int((t_final - t_initial)/step)
            pg = get_echo1_pulse_group(pi_dur, gap_2, t_initial, t_final, step, readout_start, readout, decimation, shape)
            
        case 'amplitude':
            q_duration = int_eval(params['amp_q_dur'])
            q_gap = int_eval(params['amp_q_gap'])
            a_start = int_eval(params['amp_start'])
            a_stop = int_eval(params['amp_stop'])
            step = int_eval(params['step'])
            readout_start = readout_buffer + q_gap + q_duration
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            num_patterns = int((a_stop-a_start)/step)
            
        
            
            pg = get_amp_pg(q_duration, q_gap, a_start, a_stop, step, readout_start, readout, wq_offset, phase, decimation, shape)

            
        case 'effect_temp':
            gap_2 = int_eval(params['gap_2'])
            gap_1 = int_eval(params['gap_1'])
            
            rabi_start = int_eval(params['rabi_start'])
            rabi_stop = int_eval(params['rabi_stop'])
            step = int_eval(params['step'])
            
            #pulse_1_duration = params['ge_pi_duration']
            ge_first_duration = int_eval(params['ge_first_duration'])
            ge_second_duration = int_eval(params['ge_second_duration'])
            
            wq_ge_offset = int_eval(params['ssb_freq_ge'])
            wq_ef_offset = int_eval(params['ssb_freq_ef'])
            phase_ge = int_eval(params['ssb_phase_ge'])
            phase_ef = int_eval(params['ssb_phase_ef'])
            
            readout_start = readout_buffer + ge_first_duration + gap_1 + rabi_stop + gap_1 + ge_second_duration + gap_2
            #Round to nearest multiple of decimation 
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            num_patterns = int((rabi_stop - rabi_start)/step)
            pg = get_et_pulse_group(ge_first_duration, #pulse duration
                                        ge_second_duration,
                                        gap_1,
                                        gap_2,
                                        rabi_start,
                                        rabi_stop,
                                        step,
                                        readout_start, #readout
                                        readout, #readout duration
                                        wq_ge_offset,
                                        phase_ge,
                                        wq_ef_offset,
                                        phase_ef,
                                        decimation,
                                        shape)
        case 'single_pulse':
            gap = int_eval(params['gap'])
            q_duration = int_eval(params['duration'])
           
            readout_start = gap + q_duration + readout_buffer
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            q_start = readout_start - gap - q_duration
            num_patterns = 1
            pg = get_single_pulse_group(start_time = q_start,
                                    q_duration = q_duration,
                                    readout_start = readout_start,
                                    readout = readout,
                                    frequency = wq_offset,
                                    phase = phase,
                                    #ro_freq = wr_offset,
                                    decimation = decimation,
                                    shape = shape)
        case 'sbb_freq_sweep':
            gap = params['gap']
            q_duration = params['duration']
            ssb_freq_start = params['ssb_freq_start']
            ssb_freq_end = params['ssb_freq_end']
            ssb_freq_step = params['ssb_freq_step']
            readout_start = gap + q_duration + readout_buffer
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            q_start = readout_start - gap - q_duration
            num_patterns = int((ssb_freq_end-ssb_freq_start)/ssb_freq_step)
            pg = get_sbb_freq_group(start_time = q_start,
                                    q_duration = q_duration,
                                    readout_start = readout_start,
                                    readout = readout,
                                    ssb_freq_start = ssb_freq_start,
                                    ssb_freq_end = ssb_freq_end,
                                    ssb_freq_step = ssb_freq_step,
                                    phase = phase,
                                    #ro_freq = wr_offset,
                                    decimation = decimation,
                                    shape = shape)

        case 'sbb_phase_sweep':
            gap = int_eval(params['gap'])
            q_duration = int_eval(params['duration'])
            ssb_phase_start = np.radians(params['ssb_phase_start'])
            ssb_phase_end = np.radians(params['ssb_phase_end'])
            ssb_phase_step = np.radians(params['ssb_phase_step'])
            readout_start = gap + q_duration + readout_buffer
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            q_start = readout_start - gap - q_duration
            num_patterns = int((ssb_phase_end-ssb_phase_start)/ssb_phase_step)
            pg = get_phase_pulse_group(start_time = q_start,
                                    q_duration = q_duration,
                                    readout_start = readout_start,
                                    readout = readout,
                                    frequency = wq_offset,
                                    ssb_phase_start = ssb_phase_start,
                                    ssb_phase_end = ssb_phase_end,
                                    ssb_phase_step = ssb_phase_step,
                                    #ro_freq = wr_offset,
                                    decimation = decimation,
                                    shape = shape)
    print(num_patterns)
    return pg

