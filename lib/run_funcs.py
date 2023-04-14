# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:11:27 2023

@author: lqc
"""
import sys
sys.path.append("../")
from instruments.alazar import ATS9870_NPT as npt
from . import data_process as dp
from instruments import Var_att_interface as ATT
from instruments import RF_interface as RF

import time
import numpy as np
from threading import Thread
import pyvisa as visa
from datetime import datetime
import csv
import matplotlib.pyplot as plt

#board should be acquired by running ats.Board(systemId = 1, boardId = 1)
#then npt.ConfigureBoard(board)
#awg by running be.get_awg()

#wave length should be wait_time + readout_start + readout+duration
def initialize_awg(awg, num_patterns, pattern_repeat, decimation):
    awg.set_chan_state(1, [1,2])
    
    
    for i in range(1, num_patterns):
        awg.set_seq_element_goto_state(i, 0)
    awg.set_seq_element_goto_state(num_patterns, 1)
    #set sampling rate
    new_freq = 1/decimation
    awg.set_freq(str(new_freq)+"GHZ")
    
    #delay = pattern_repeat * num_patterns * wave_len # #this in ns convert to seconds
    #delay /= 1e9
    #delay *= 100
    #add delay margin (could be less)
    #delay *= 1.1
    #awg.set_trig_source('internal')
    #time.sleep(.2)
    #awg.set_trig_interval(delay)
    #time.sleep(.2)

#This function sets the static parameters before running
def init_params(params):
    rm = visa.ResourceManager()
    q_rf = RF.RF_source(rm, "TCPIP0::172.20.1.7::5025::SOCKET")
    r_rf = RF.RF_source(rm, "TCPIP0::172.20.1.8::5025::SOCKET")
    q_att = ATT.Atten(rm, "TCPIP0::172.20.1.6::5025::SOCKET")
    r_att = ATT.Atten(rm, "TCPIP0::172.20.1.9::5025::SOCKET")
    GHz = 1e9
    q_rf.set_freq(params['set_wq']*GHz)
    r_rf.set_freq(params['set_wr']*GHz)
    q_rf.set_power(params['set_pq'])
    r_rf.set_power(params['set_pr'])
    q_att.set_attenuation(params['set_q_att'])
    r_att.set_attenuation(params['set_r_att'])

def run_and_acquire(awg,
                board,
                params,
                num_patterns,
                save_raw,
                path):
    """
    runs sequence on AWG once. params should be dictionary of YAML file.
    """
    samples_per_ac = params['acq_multiples']*256
    pattern_repeat = params['pattern_repeat']
    seq_repeat = params['seq_repeat']
    
    acproc = Thread(target = npt.AcquireData, args = (board, params, num_patterns, path, save_raw, False))
    acproc.start()
    time.sleep(.3)
    awg.run()
    acproc.join()
    
    awg.stop()
    
    #DATA processing part
    chA = None
    chB = None
    if save_raw:
        (chA, chB) = dp.frombin(tot_samples = samples_per_ac, numAcquisitions = num_patterns*pattern_repeat*seq_repeat, channels = 2, name = path + "rawdata.bin")
    
    return (chA, chB)
    
    #dp.plot_all(chA, chB, 1, pattern_repeat, seq_repeat, large_data_plot = large_plot)
    
    
#this function will take one of the do_all functions as a parameter
#sweep param should be 'wq', 'pq', 'wr', 'pr', 'att'
#w is for frequency, p for power
#q for qubit, r for readout
#other instrument(s) will not be changed


#extra_column used by double sweep function to store second parameter.
#It should be python list of ["parameter name", value]
def single_sweep(name, awg, board, num_patterns, params, sweep_param, start, stop, step, avg_start, avg_length, extra_column = None, live_plot = False):
    #1.8 rf is for qubit
    #1.7 rf is for readout
    rm = visa.ResourceManager()
    qubit_addr = "TCPIP0::172.20.1.7::5025::SOCKET"
    readout_addr = "TCPIP0::172.20.1.8::5025::SOCKET"
    q_atten_addr = "TCPIP0::172.20.1.6::5025::SOCKET"
    r_atten_addr = "TCPIP0::172.20.1.9::5025::SOCKET"

    addrs = {
        'wr': readout_addr,
        'wq': qubit_addr,
        'pr': readout_addr,
        'pq': qubit_addr
    }

    if sweep_param == 'q_att':
        inst = ATT.Atten(rm, q_atten_addr)
    elif sweep_param == 'r_att':
        inst = ATT.Atten(rm, r_atten_addr)
    else:
        inst = RF.RF_source(rm, addrs[sweep_param])


    if sweep_param == 'wq' or sweep_param == 'wr':
        func_call = inst.set_freq
    elif sweep_param == 'pq' or sweep_param == 'pr':
        func_call = inst.set_power
    else:
        #attenuator
        func_call = inst.set_attenuation

    #rf.write(':OUTPut:STATe ON')

    avgsA_sub = np.zeros((num_patterns, len(np.arange(start,stop,step))))
    avgsB_sub = np.zeros((num_patterns, len(np.arange(start,stop,step))))

    avgsA_nosub = np.zeros((num_patterns, len(np.arange(start,stop,step))))
    avgsB_nosub = np.zeros((num_patterns, len(np.arange(start,stop,step))))


    #h, = plt.plot(np.random.randn(50))
    #plt.draw()

    sweep_num = 0
    for param in np.arange(start, stop, step):
        #this is the func call that sets the new sweep parameter built from previous commands
        func_call(param)
        #print(func_call)
        time.sleep(.05)
        
        run_and_acquire(awg,
                        board,
                        params,
                        num_patterns,
                        save_raw = False,
                        path = name)
        
        #run_and_acquire(awg, 
        #                board,
        #                seq_repeat = seq_repeat,
        #                pattern_repeat = pattern_repeat,
        #                num_patterns = num_patterns,
        #                samples_per_ac=samples_per_ac,
        #                avg_start=avg_start,
        #                avg_length=avg_length,
        #                save_raw=False,
        #                path = name)
        
        
        #cAp = dp.get_p1_p2(chA, num_patterns, pattern_repeat, seq_repeat)
        #cBp = dp.get_p1_p2(chB, num_patterns, pattern_repeat, seq_repeat)
        #print(np.shape(cAp))

        #this plots individual sweeps
        
        
        chA_sub = np.load(name + "chA_sub.npy")
        chB_sub = np.load(name + "chB_sub.npy")
        chA_nosub = np.load(name + "chA_nosub.npy")
        chB_nosub = np.load(name + "chB_nosub.npy")
        
        
        '''
        plot1 = dp.average_all_iterations(cAp[0])
        plot2 = dp.average_all_iterations(cAp[1])
        ts = time.time()
        plt.plot(plot1, color = 'red')
        plt.plot(plot2, color = 'green')
        plt.title('chA '+str(param))
        plt.figure()
        te = time.time()
        print(te-ts)
        plot1 = dp.average_all_iterations(cBp[0])
        plot2 = dp.average_all_iterations(cBp[1])
        plt.plot(plot1, color = 'red')
        plt.plot(plot2, color = 'green')
        plt.title('chB '+str(param))
        plt.figure()
        '''
        
        #avgsA should be array of shape(num_patterns, sweep_num, x)
        
        for i in range(num_patterns):
            avgsA_sub[i][sweep_num] = np.average(chA_sub[i])
            avgsB_sub[i][sweep_num] = np.average(chB_sub[i])
            avgsA_nosub[i][sweep_num] = np.average(chA_nosub[i])
            avgsB_nosub[i][sweep_num] = np.average(chB_nosub[i])
            
            #can probably do csv saving here
            #maybe make csv files again, with columns being [channel A, channel B, pattern#, sweep_param_val]
            #assume name is the path without file name
            #name of file will be sweepparam_sweepval_pattern#.csv
            #pattern num = i
            #sweep param val = param
            #channel A = avgsA[i][sweep_num]
            
            f_name = sweep_param + "_" + str(param) + "_" + str(i) + ".csv"
            with open(name + '_' + f_name, 'w', newline='') as output:
                wr = csv.writer(output, delimiter=',', quoting=csv.QUOTE_NONE)
                if extra_column != None:
                    header = header=['ChA','ChB','pattern_num', str(sweep_param), extra_column[0]]
                else:
                    header=['ChA','ChB','pattern_num', str(sweep_param)]
                wr.writerow(header)
                
                
                #t_avgA = dp.average_all_iterations(cAp[i])
                #t_avgB = dp.average_all_iterations(cBp[i])
                
                #for j in range(len(t_avgA)):
                #    if extra_column != None:
                #        wr.writerow([t_avgA[j], t_avgB[j], i, param, extra_column[1]])
                #    else:    
                #        wr.writerow([t_avgA[j], t_avgB[j], i, param])
        
        sweep_num += 1
        
        
    #returns an array of sweep averages
    #avgsA[i] is pattern i
    #avgsA[i][j] is the average of all averages of the jth sweep, of pattern i
    return (avgsA_sub, avgsB_sub, avgsA_nosub, avgsB_nosub)
    
    
def double_sweep(name,
                 awg,
                 board,
                 param1,
                 p1start,
                 p1stop,
                 p1step,
                 param2,
                 p2start,
                 p2stop,
                 p2step,
                 samp_per_ac,
                 num_patterns,
                 pattern_repeat,
                 seq_repeat,
                 wlen,
                 avg_start,
                 avg_length,
                 live_plot = False):

    rm = visa.ResourceManager()
    #ATT = rm.open_resource('TCPIP0::172.20.1.6::5025::SOCKET')
    #command = ":ATT "+str(pstart)+"dB"

    if param1 == 'wq' or param1 == 'pq':
        #qubit rf
        inst = rm.open_resource('TCPIP0::172.20.1.7::5025::SOCKET')
    elif param1 == 'wr' or param1 == 'pr':
        #readout rf
        inst = rm.open_resource('TCPIP0::172.20.1.8::5025::SOCKET')
    else:
        #attenuator
        inst = rm.open_resource('TCPIP0::172.20.1.6::5025::SOCKET')

    if param1 == 'wq' or param1 == 'wr':
        command_pre = ':FREQuency:CW '
        unit = 'Hz'
    elif param1 == 'pq' or param1 == 'pr':
        command_pre = ':POWer:AMPLitude '
        unit = 'DBM'
    else:
        #attenuator
        #command = ":ATT "+str(pstart)+"dB"
        command_pre = ':ATT '
        unit = 'dB'
    #command = ':FREQuency:CW '+ str(pstart) + "Hz"
    #command = ":ATT "+str(pstart)+"dB"

    #inner loop is p2, outer loop is p1

    ylen = len(np.arange(p1start, p1stop, p1step))
    xlen = len(np.arange(p2start, p2stop, p2step))
    
    finalA = np.zeros((num_patterns, ylen, xlen))
    finalB = np.zeros((num_patterns, ylen, xlen))
    
    
    if live_plot:
        plt.ion()
        fig = plt.figure()
        
    
    
    sweep_num = 0

    for new_param in np.arange(p1start, p1stop, p1step):
        t_name = name + "_" + param1 + "_" + str(new_param)
        command = command_pre + str(new_param) + unit
        print(command)
        inst.write(command)
        time.sleep(.005)
    
        avgsA, avgsB = single_sweep(t_name,
                                    awg,
                                    board,
                                    num_patterns,
                                    pattern_repeat,
                                    seq_repeat,
                                    samp_per_ac,
                                    wlen,
                                    param2,
                                    p2start,
                                    p2stop,
                                    p2step,
                                    avg_start,
                                    avg_length,
                                    extra_column = [param1, new_param])

        #avgsA has shape [num_patterns][num_sweeps]
        #finalA should have shape [num_patterns][sweepl1][sweepl2]
        finalA[:, sweep_num] = avgsA[:]
        finalB[:, sweep_num] = avgsB[:]

        sweep_num += 1


    inst.write(':OUTPut:STATe OFF')

    return finalA, finalB
    
    
