

import sys
sys.path.append("../")

#import npp_funcs as nppf
from lib import wave_construction as be
import yaml
from lib import run_funcs
import math
import numpy as np

def get_nopi_pi_group(
                start_time, #pulse start time
                q_duration, #pulse duration
                readout_start, #readout
                readout, #readout duration
                frequency,
                #ro_freq,
                decimation):
    
    start_time = int(start_time/decimation)
    q_duration = int(q_duration/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    frequency *= decimation
    
    #start, duration, amplitude, channel, sweep_type, sweep_end, sweep_step, readout
    p1 = be.Sweep_Pulse(start_time, q_duration, amplitude = 0, frequency=frequency, channel = 1, sweep_param = 'amplitude', sweep_stop = 1.1, sweep_step = 1)
    #ro = be.Sin_Readout(readout_start, readout, amplitude = 1, frequency=ro_freq)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([p1, ro])
    return pg

def get_gaussian_pulse_group(peak ,
                             sigma,
                             mu,
                             gap,
                             readout_start,
                             readout,
                             freq,
                             numsig,
                             decimation):
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    p1 = be.gaussian(mu,peak,gap,sigma,freq,numsig,channel = 1)
    pg = be.PulseGroup([p1, ro])
    return pg

def get_gaussian_sweep_pulse_group(peak,
                                   initialstartpoint,
                                   finalstartpoint,
                                   initial_duration,
                                   final_duration,
                                   step,
                                   freq,
                                   totalsig,
                                   readout_start,
                                   readout,
                                   sweep_param,
                                   decimation):
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    p1 = be.sweep_gaussian(peak,
                           initialstartpoint,
                           finalstartpoint,
                           initial_duration,
                           final_duration,
                           step,freq,
                           sweep_param,
                           totalsig,
                           channel = 1)
    pg = be.PulseGroup([p1, ro])
    return pg

def get_gaussian_amp_sweep_pulse_group(initial_amp, final_amp, step, duration, freq, totalsig, mu,
                                       readout_start, readout, decimation):
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    p1 = be.amp_sweep_gaussian(initial_amp,
                                   final_amp,
                                   step,
                                   duration,
                                   freq,
                                   totalsig,
                                   mu,
                                   channel = 1)
    pg = be.PulseGroup([p1, ro])
    return pg

def get_gaussian_gap_sweep_pulse_group(amplitude,mu0, muf, step, duration, freq, totalsig,
                                        readout_start, readout, decimation):
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    p1 = be.gap_sweep_gaussian(amplitude,
                                mu0,
                                muf,
                                step,
                                duration,
                                freq,
                                totalsig,
                                channel = 1)
    pg = be.PulseGroup([p1, ro])
    return pg

def get_readout_group(
                readout_start, #readout
                readout, #readout duration
                #freq,
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
                decimation = 1):
    
    q_dur = int(q_dur/decimation)
    q_start_start = int(q_start_start/decimation)
    q_start_end = int(q_start_end/decimation)
    q_start_step = int(q_start_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    frequency *= decimation
    
    p1 = be.Sweep_Pulse(q_start_start, q_dur, amplitude = 1, frequency=frequency, channel = 1, sweep_param = 'start', sweep_stop = q_start_end, sweep_step = q_start_step)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([p1, ro])
    #pg.send_waves_awg(awg, "hi", 5)
    return pg




def get_echo_pulse_group(pi_dur, 
                gap_2,
                t_start,
                t_stop,
                t_step,
                readout_start, #readout
                readout, #readout duration
                frequency,
                decimation):
    pi_dur = int(pi_dur/decimation)
    gap_2 = int(gap_2/decimation)
    t_start = int(t_start/decimation)
    t_stop = int(t_stop/decimation)
    t_step = int(t_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    frequency *= decimation
    
    x_pulse_2 = be.Pulse(readout_start - gap_2 - int(pi_dur/2) , int(pi_dur/2), 1, 1)
    
    
    first_pulse_start = readout_start - gap_2 - (t_stop) - (2*pi_dur)# - t_step
    first_pulse_end = readout_start - gap_2 - (t_start) - (2*pi_dur) + t_step
    
    x_pulse_1 = be.Sweep_Pulse(first_pulse_start, int(pi_dur/2), amplitude = 1, frequency = frequency, channel = 1, 
                        sweep_param = 'start',
                        sweep_stop = first_pulse_end,
                        sweep_step = t_step)
    
    
    p2_start = readout_start - gap_2 - int(t_stop/2) - int(3*pi_dur/2) #- int(t_step/2)
    p2_end = readout_start - gap_2 - int(t_start/2) - int(3*pi_dur/2) + int(t_step/2)
    #p2_end += half_t_step
    
    y_pulse = be.Sweep_Pulse(p2_start, pi_dur, amplitude = 1, frequency = frequency, channel = 2, sweep_param = 'start', sweep_stop = p2_end, sweep_step = int(t_step/2))
    
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
                frequency,
                decimation = 1):
    pi_dur = int(pi_dur/decimation)
    gap_2 = int(gap_2/decimation)
    t_start = int(t_start/decimation)
    t_stop = int(t_stop/decimation)
    t_step = int(t_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    frequency *= decimation
    
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
    
    y_pulse = be.Sweep_Pulse(p2_start, pi_dur, amplitude = 1, frequency=frequency, channel = 1, sweep_param = 'start', sweep_stop = p2_end, sweep_step = int(t_step/2))
    
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
                decimation = 1):
    
    start_time = int(start_time/decimation)
    q_dur_start = int(q_dur_start/decimation)
    q_dur_stop = int(q_dur_stop/decimation)
    q_dur_step = int(q_dur_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    frequency *= decimation
    
    
    p1 = be.Sweep_Pulse(start_time, q_dur_start, amplitude = 1, frequency=frequency, channel = 1, sweep_param = 'duration', sweep_stop = q_dur_stop, sweep_step = q_dur_step)
    
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
                frequency,
                decimation = 1):
    q_duration = int(q_duration/decimation)
    first_pulse_start = int(first_pulse_start/decimation)
    second_pulse_start = int(second_pulse_start/decimation)
    first_pulse_final_start = int(first_pulse_final_start/decimation)
    first_pulse_step = int(first_pulse_step/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    frequency *= decimation
    
    p1 = be.Sweep_Pulse(first_pulse_start, q_duration, amplitude = 1,
                        frequency=frequency,
                        channel = 1,
                        sweep_param = 'start',
                        sweep_stop = first_pulse_final_start,
                        sweep_step = first_pulse_step)
    p2 = be.Sin_Pulse(second_pulse_start, q_duration, amplitude = 1, frequency=frequency, channel = 1)
    ro = be.Readout_Pulse(readout_start, readout, amplitude = 1)
    pg = be.PulseGroup([p1, p2, ro])
    return pg

def get_amp_pg(q_duration,
               q_gap,
               amp_start,
               amp_stop,
               amp_step,
               readout_start,
               readout,
               frequency,
               decimation = 1):
    q_duration = int(q_duration/decimation)
    q_gap = int(q_gap/decimation)
    readout_start = int(readout_start/decimation)
    readout = int(readout/decimation)
    
    q_start = readout_start - q_gap - q_duration
    p1 = be.Sweep_Pulse(q_start, q_duration, amp_start, frequency, 'amp', amp_stop + amp_step, amp_step)
    
    ro = be.Readout_Pulse(readout_start, readout, 1)
    pg = be.PulseGroup([p1, ro])
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
                       frequency,
                       decimation):
    
    
    p1_start_init = readout_start - gap_2 - rabi_start - gap_1 - ge_first_duration - gap_1 - ge_second_duration
    p1_start_final = readout_start - gap_2 - rabi_stop - gap_1 - ge_first_duration  - gap_1 - ge_second_duration
    
    p1_start_init += rabi_step
    p1_start_final -= rabi_step
    #p1 has to be shifted by one step because duration sweeps from left to right
    #and start time sweeps from right to left.
    #This may be nice to fix in wave construction of sweep pulse.
    
    
    #self, start, duration, amplitude, frequency, sweep_param, sweep_stop, sweep_step, channel = None):
    pulse1 = be.Sweep_Pulse(p1_start_final,
                            ge_first_duration,
                            amplitude = 1,
                            frequency = 0,
                            sweep_param = 'start',
                            sweep_stop = p1_start_init,
                            sweep_step = rabi_step,
                            channel = 1
                            )
    
    p2_start_init = readout_start - gap_2 - rabi_start - gap_1 - ge_second_duration
    
    pulse2 = be.Sweep_Pulse(p2_start_init,
                            rabi_start,
                            amplitude = 1,
                            frequency = 0,
                            sweep_param = 'duration',
                            sweep_stop = rabi_stop,
                            sweep_step = rabi_step,
                            phase = np.pi/2,
                            channel = 1
                            )
    
    #self, start: int, duration: int, amplitude: float, frequency: float, channel: int
    p3_start = readout_start - gap_2 - ge_second_duration
    
    pulse3 = be.Sin_Pulse(p3_start, ge_second_duration, amplitude = 1, frequency = 0, channel = 1)
    ro = be.Readout_Pulse(readout_start, readout_dur, 1)
    pg = be.PulseGroup([pulse1, pulse2, pulse3, ro])
    #pg = be.PulseGroup([pulse2, pulse3, ro])
    
    return pg

def get_pg(params):

    #First get params that are used by all measurements (readout values, name)
    #measurement will describe which pulse to send
    measurement = params['measurement']
    readout_buffer = int(params['readout_trigger_offset'] * 1.5)
    #readout_buffer = 0
    decimation = params['decimation']
    #readout_start = params['readout_start']
    
    readout_trigger_offset = params['readout_trigger_offset']
    wq_offset = params['set_wq_offset']
    #wr_offset = params['set_wr_offset']
    params = params[measurement]
    readout = params['readout_duration']

    match measurement:
        case 'gaussian sweep (gap)':
            amplitude = params['amplitude']
            initial_gap = params['initial gap']
            final_gap = params['final gap']
            step = params['steps']
            duration = params['duration']
            freq = params['frequency']

            totalsig = 6
            readout_start = duration + final_gap + readout_buffer
            mu0= readout_start - duration/2 - initial_gap
            muf= readout_start - duration/2 - final_gap
            num_patterns = (final_amp - initial_amp)/step
            pg = get_gaussian_gap_sweep_pulse_group(amplitude,mu0, muf, step, duration, freq, totalsig,
                                                     readout_start, readout, decimation)
            
        case 'gaussian sweep (amplitude)':
            initial_amp = params['initial amplitude']
            final_amp = params['final amplitude']
            step = params['steps']
            gap = params['gap']
            duration = params['duration']
            freq = params['frequency']

            totalsig = 6
            readout_start = duration + gap + readout_buffer
            mu = readout_start - duration/2 - gap
            num_patterns = (final_amp - initial_amp)/step
            pg = get_gaussian_amp_sweep_pulse_group(initial_amp, final_amp, step, duration, freq, totalsig, mu,
                                                     readout_start, readout, decimation)

        case 'gaussian sweep (sigma)':
            peak = params['amplitude']
            gap = params['gap']
            initial_duration = params['initial duration']
            final_duration = params['final duration']
            step = params['steps']
            freq = params['frequency']
            
            totalsig=6
            sweep_param = "sigma"
            readout_start = final_duration + gap + readout_buffer
            initialstartpoint = readout_start - gap - initial_duration
            finalstartpoint = readout_start - gap - final_duration

            sigma0=initial_duration/totalsig
            sigmaf=final_duration/totalsig
            num_patterns = (sigmaf-sigma0)/step

            pg = get_gaussian_sweep_pulse_group(peak,initialstartpoint,finalstartpoint,initial_duration,final_duration,
                                                step,freq,totalsig,readout_start,readout,sweep_param,decimation)

        case 'gaussian':
            peak = params['amplitude']
            #sigma = params['sigma']
            gap = params['gap']
            duration = params['duration']
            freq = params['frequency']
            totalsig=6
            numsig=totalsig/2
            sigma = duration/(2*numsig)
            readout_start = duration + gap + readout_buffer
            mu = readout_start - duration/2 - gap
            num_patterns = 1

            pg = get_gaussian_pulse_group(peak,sigma,mu,gap,
                                          readout_start,readout,freq,numsig,decimation)

        case 'T1':
            q_duration = params['T1_q_dur']
            init_gap = params['T1_init_gap']
            final_gap = params['T1_final_gap']
            step = params['step']
            readout_start = q_duration + final_gap + readout_buffer
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            q_start_start = readout_start - final_gap - q_duration
            q_start_end = readout_start - init_gap - q_duration + step
            num_patterns = int((q_start_end - q_start_start)/step)
            pg = get_T1_pulse_group(q_duration, q_start_start, q_start_end, step, readout_start, readout, wq_offset, decimation)

        case 'rabi':
            step = params['step']
            q_dur_start = params['rabi_pulse_initial_duration']
            q_dur_stop = params['rabi_pulse_end_duration'] + step
            gap = params['rabi_pulse_gap']
            
            readout_start = gap + q_dur_stop + readout_buffer
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            start_time = readout_start - gap - q_dur_start
            num_patterns = int((q_dur_stop - q_dur_start)/step)
            pg = get_rabi_pulse_group(start_time, q_dur_start, q_dur_stop, step, readout_start, readout, wq_offset, decimation)

        case 'ramsey':
            gap2 = params['ramsey_gap_2']
            g1_init = params['ramsey_gap_1_init']
            g1_final = params['ramsey_gap_1_final']
            step = params['step']
            q_duration = params['ramsey_q_dur'] # pi/2 pulse
            
            
            
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
                                        decimation)

        case 'npp':
            gap = params['nop_p_q_gap']
            q_duration = params['nop_p_q_dur']
            readout_start = gap + q_duration + readout_buffer
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            q_start = readout_start - gap - q_duration
            num_patterns = 2
            pg = get_nopi_pi_group(start_time = q_start,
                                    q_duration = q_duration,
                                    readout_start = readout_start,
                                    readout = readout,
                                    frequency = wq_offset,
                                    #ro_freq = wr_offset,
                                    decimation = decimation)

        case 'readout':
            num_patterns = 1
            readout_start = readout_buffer
            readout_start = decimation * math.ceil(readout_start/decimation)
            pg = get_readout_group(readout_start = readout_start,
                            readout = readout,
                            #freq = wr_offset,
                            decimation = decimation)
            

        case 'echo':
            gap_2 = params['echo_gap_2']
            t_initial = params['echo_initial_t']
            t_final = params['echo_final_t']
            pi_dur = params['echo_pi_pulse']
            step = params['step']
            
            readout_start = readout_buffer + 2*pi_dur + t_final + gap_2
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            num_patterns = int((t_final - t_initial)/step)
            pg = get_echo_pulse_group(pi_dur, gap_2, t_initial, t_final, step, readout_start, readout, wq_offset, decimation)

        case 'echo_1ax':
            gap_2 = params['echo_gap_2']
            t_initial = params['echo_initial_t']
            t_final = params['echo_final_t']
            pi_dur = params['echo_pi_pulse']
            step = params['step']
            readout_start = readout_buffer + 2*pi_dur + t_final + gap_2
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            num_patterns = int((t_final - t_initial)/step)
            pg = get_echo1_pulse_group(pi_dur, gap_2, t_initial, t_final, step, readout_start, readout, wq_offset, decimation)
            
        case 'amplitude':
            q_duration = params['amp_q_dur']
            q_gap = params['amp_q_gap']
            a_start = params['amp_start']
            a_stop = params['amp_stop']
            step = params['step']
            readout_start = readout_buffer + q_gap + q_duration
            readout_start = decimation * math.ceil(readout_start/decimation)
            
            num_patterns = int((a_stop-a_start)/step)
            pg = get_amp_pg(q_duration, q_gap, a_start, a_stop, step, readout_start, wq_offset, readout)
            
        case 'effect_temp':
            gap_2 = params['gap_2']
            gap_1 = params['gap_1']
            
            rabi_start = params['rabi_start']
            rabi_stop = params['rabi_stop']
            step = params['step']
            
            #pulse_1_duration = params['ge_pi_duration']
            ge_first_duration = params['ge_first_duration']
            ge_second_duration = params['ge_second_duration']
            
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
                                        wq_offset,
                                        decimation)

    print(num_patterns)
    return pg

