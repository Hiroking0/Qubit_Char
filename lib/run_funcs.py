# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:11:27 2023

@author: lqc
"""
import sys
sys.path.append("../")
from instruments.alazar import ATS9870_NPT as npt
from instruments import Var_att_interface as ATT
from instruments import RF_interface as RF
from instruments.TekAwg import tek_awg as tawg

import time
import numpy as np
from threading import Thread
import pyvisa as visa
import csv
import matplotlib.pyplot as plt
import queue
from datetime import datetime
import pickle as pkl

class Data_Arrs:
    def __init__(self, *args): #a_nosub, a_sub, b_nosub, b_sub, mags_nosub, mags_sub):
        self.a_nosub = args[0]
        self.a_sub = args[1]
        self.b_nosub = args[2]
        self.b_sub = args[3]
        self.mags_nosub = args[4]
        self.mags_sub = args[5]
        self.readout_A = args[6]
        self.readout_B = args[7]

    def save(self, path, name):
        now = datetime.now()
        date = now.strftime("%m%d_%H%M%S")
        with open(name + "_" + date, 'wb') as pickle_file:
            pkl.dump(self, pickle_file)

    def get_data_arrs(self):
        return (self.a_nosub, self.a_sub, self.b_nosub, self.b_sub, self.mags_nosub, self.mags_sub, self.readout_A, self.readout_B)
    
    def get_avgs(self):
        pass
        '''
        for i in range(num_patterns):
        pattern_avgs_cA[i] = np.average(chA_nosub[i])
        pattern_avgs_cB[i] = np.average(chB_nosub[i])
        mags[i] = np.average(mags_nosub[i])
        #mags[i] = np.average(np.sqrt(chB_nosub[i] ** 2 + chA_nosub[i] ** 2))
        
        pattern_avgs_cA_sub[i] = np.average(chA_sub[i])
        pattern_avgs_cB_sub[i] = np.average(chB_sub[i])
        mags_sub[i] = np.average(mags_sub[i])
        #mags_sub[i] = np.average(np.sqrt(chB_sub[i] ** 2 + chA_sub[i] ** 2))
        '''
    def gen_data_arrs(self):
        yield self.a_nosub
        yield self.a_sub
        yield self.b_nosub
        yield self.b_sub
        yield self.mags_nosub
        yield self.mags_sub
        
        



def initialize_awg(awg, num_patterns, pattern_repeat, decimation):
    awg.set_chan_state(1, [1,2,3,4])    
    
    for i in range(1, num_patterns):
        awg.set_seq_element_goto_state(i, 0)
        awg.set_seq_element_loop_cnt(i, pattern_repeat)
    awg.set_seq_element_goto_state(num_patterns, 1)
    #set sampling rate
    new_freq = 1/decimation
    awg.set_freq(str(new_freq)+"GHZ")


#This function sets the static parameters before running
def init_params(params):
    rm = visa.ResourceManager()
    q_rf = RF.RF_source(rm, "TCPIP0::172.20.1.7::5025::SOCKET")
    r_rf = RF.RF_source(rm, "TCPIP0::172.20.1.8::5025::SOCKET")
    q_att = ATT.Atten(rm, "TCPIP0::172.20.1.6::5025::SOCKET")
    r_att = ATT.Atten(rm, "TCPIP0::172.20.1.9::5025::SOCKET")
    twpa_rf = RF.RF_source(rm, "TCPIP0::172.20.1.11::5025::SOCKET")
    q_rf.set_freq(params['set_wq'])
    r_rf.set_freq(params['set_wr'])
    q_rf.set_power(params['set_pq'])
    r_rf.set_power(params['set_pr'])
    q_att.set_attenuation(params['set_q_att'])
    r_att.set_attenuation(params['set_r_att'])
    twpa_rf.set_power(params['set_p_twpa'])
    twpa_rf.set_freq(params['set_w_twpa'])
    #q_rf.enable_out()
    #r_rf.enable_out()
    
def run_and_acquire(awg,
                board,
                params,
                num_patterns,
                path):
    """
    runs sequence on AWG once. params should be dictionary of YAML file.
    """

    save_raw = False
    live_plot = False
    que = queue.Queue()

    acproc = Thread(target = lambda q, board, params, num_patterns, path, raw, live:
                            q.put(npt.AcquireData(board, params, num_patterns, path, raw, live)), 
                            args = (que, board, params, num_patterns, path, save_raw, live_plot))

    acproc.start()
    time.sleep(.3)
    awg.run()
    acproc.join()
    awg.stop()
    
    data_arrs = Data_Arrs(que.get())

    return data_arrs
    
    
#this function will take one of the do_all functions as a parameter
#sweep param should be 'wq', 'pq', 'wr', 'pr', 'att'
#w is for frequency, p for power
#q for qubit, r for readout
#other instrument(s) will not be changed



def get_func_call(rm, sweep_param, awg):
    qubit_addr = "TCPIP0::172.20.1.7::5025::SOCKET"
    readout_addr = "TCPIP0::172.20.1.8::5025::SOCKET"
    q_atten_addr = "TCPIP0::172.20.1.6::5025::SOCKET"
    r_atten_addr = "TCPIP0::172.20.1.9::5025::SOCKET"
    twpa_addr = "TCPIP0::172.20.1.11::5025::SOCKET"
    #amp_addr = "TCPIP::129.2.108.72::hislip0,4880::INSTR"
    
    addr_table = {
        'wr': readout_addr,
        'wq': qubit_addr,
        'pr': readout_addr,
        'pq': qubit_addr,
        'r_att': r_atten_addr,
        'q_att': q_atten_addr,
        'p_twpa': twpa_addr,
        'w_twpa': twpa_addr,
        #'amp':amp_addr
    }
    
    inst_table = {
        'wr': RF.RF_source,
        'wq': RF.RF_source,
        'pq': RF.RF_source,
        'pr': RF.RF_source,
        'r_att': ATT.Atten,
        'q_att': ATT.Atten,
        #'amp': tawg.connect_raw_visa_socket,
        }
    if sweep_param == "amp":
        inst = awg
    else:
        inst_class = inst_table[sweep_param]
        address = addr_table[sweep_param]
        inst = inst_class(rm, address)

    func_table = {
        'wq': 'set_freq',
        'wr': 'set_freq',
        'pr': 'set_power',
        'pq': 'set_power',
        'r_att': 'set_attenuation',
        'q_att': 'set_attenuation',
        'amp' : "set_amplitude"
        }
    
    func_call = getattr(inst, func_table[sweep_param])
    return func_call


#extra_column used by double sweep function to store second parameter.
#It should be python list of ["parameter name", value]
def single_sweep(name,
                 awg,
                 board,
                 num_patterns,
                 params,
                 extra_column = None,
                 live_plot = False):
    #1.8 rf is for qubit
    #1.7 rf is for readout
    

    att_sweeps = ['q_att', 'r_att']
    rf_sweeps = ['wq', 'pq', 'wr', 'pr', 'w_twpa', 'p_twpa']
    ena_sweeps = ['w_ena', 'p_ena']
    
    freq_sweeps = ['wq', 'wr', 'w_twpa', 'w_ena']
    pow_sweeps = ['pq', 'pr', 'p_twpa', 'p_ena']

    sweep_param = params['p1']
    start = params['p1start']
    stop = params['p1stop']
    step = params['p1step']
    
    rm = visa.ResourceManager()
    func_call = get_func_call(rm, sweep_param, awg)

    avgsA_sub = np.zeros((num_patterns, len(np.arange(start,stop,step))))
    avgsB_sub = np.zeros((num_patterns, len(np.arange(start,stop,step))))

    avgsA_nosub = np.zeros((num_patterns, len(np.arange(start,stop,step))))
    avgsB_nosub = np.zeros((num_patterns, len(np.arange(start,stop,step))))

    mags_sub = np.zeros((num_patterns, len(np.arange(start,stop,step))))
    mags_nosub = np.zeros((num_patterns, len(np.arange(start,stop,step))))
    #print(np.shape(mags_nosub))

    if live_plot:
        plt.ion()
        
        fig, ax1 = plt.subplots()
        #ax = fig.add_subplot(111)
        #plt.pcolormesh(mags_nosub)
        #fig.canvas.draw()
        axim1 = ax1.imshow(mags_nosub, vmin=280, vmax=320)
        
        #myobj = plt.imshow(mags_nosub, vmin = 100, vmax = 400)

    sweep_num = 0
    sweeps = np.arange(start, stop, step)
    for param in sweeps:
        #this is the func call that sets the new sweep parameter built from previous commands
        func_call(param)
        #print(func_call)
        time.sleep(.05)
        
        (chA_sub, chB_sub, chA_nosub, chB_nosub, mag_sub, mag_nosub) = run_and_acquire(awg,
                                                                                       board,
                                                                                       params,
                                                                                       num_patterns,
                                                                                       save_raw = False,
                                                                                       path = name)
        
        #avgsA should be array of shape(num_patterns, sweep_num, x)
        
        for i in range(num_patterns):
            avgsA_sub[i][sweep_num] = np.average(chA_sub[i])
            avgsB_sub[i][sweep_num] = np.average(chB_sub[i])
            avgsA_nosub[i][sweep_num] = np.average(chA_nosub[i])
            avgsB_nosub[i][sweep_num] = np.average(chB_nosub[i])
            mags_sub[i][sweep_num] = np.average(mag_sub[i])
            mags_nosub[i][sweep_num] = np.average(mag_nosub[i])
        #Here avgs[:][0:sweep_num] should be correct. the rest of avgs[:][sweep_num:] should be 0
        
        if live_plot and sweep_num > 0:
            axim1.set_data(mags_nosub)
            fig.canvas.flush_events()
            plt.pause(.01)
            
        
        sweep_num += 1
            
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
            header = header=['chA_sub','chB_sub','mag_sub', 'chA_nosub', 'chB_nosub', 'mag_nosub', 'pattern_num', str(sweep_param), extra_column[0]]
        else:
            header=['chA_sub','chB_sub','mag_sub', 'chA_nosub', 'chB_nosub', 'mag_nosub', 'pattern_num', str(sweep_param)]
        wr.writerow(header)
        
        for pattern in range(len(avgsA_sub)):
            for j in range(len(avgsA_nosub[pattern])):
                #for each sweep, write the row into the file
                t_row = [avgsA_sub[pattern][j],
                         avgsB_sub[pattern][j],
                         mags_sub[pattern][j],
                         avgsA_nosub[pattern][j],
                         avgsB_nosub[pattern][j],
                         mags_nosub[pattern][j],
                         pattern,
                         sweeps[j]
                         ]
                if extra_column != None:
                    t_row.append(extra_column[1])
                
                wr.writerow(t_row)
    return (avgsA_sub, avgsB_sub, avgsA_nosub, avgsB_nosub, mags_sub, mags_nosub)
    
    
    

    
    
def double_sweep(name,
                 awg,
                 board,
                 params,
                 num_patterns,
                 live_plot = False):

    rm = visa.ResourceManager()
    
    p1start = params['p1start']
    p1stop = params['p1stop']
    p1step = params['p1step']
    
    param2 = params['p2']
    p2start = params['p2start']
    p2stop = params['p2stop']
    p2step = params['p2step']
    
    #inner loop is p2, outer loop is p1

    rm = visa.ResourceManager()
    func_call = get_func_call(rm, param2, awg)

    ylen = len(np.arange(p1start, p1stop, p1step))
    xlen = len(np.arange(p2start, p2stop, p2step))
    
    f_A_nosub = np.zeros((num_patterns, ylen, xlen))
    f_B_nosub = np.zeros((num_patterns, ylen, xlen))
    f_A_sub = np.zeros((num_patterns, ylen, xlen))
    f_B_sub = np.zeros((num_patterns, ylen, xlen))
    f_M_nosub = np.zeros((num_patterns, ylen, xlen))
    f_M_sub = np.zeros((num_patterns, ylen, xlen))
    
    if live_plot:
        plt.ion()
        fig = plt.figure()
        
    sweep_num = 0

    for new_param in np.arange(p2start, p2stop, p2step):
        t_name = name + "_" + param2 + "_" + str(new_param)
        #command = command_pre + str(new_param) + unit
        #print(command)
        #inst.write(command)
        print("outer param:", new_param)
        func_call(new_param)
    
    
    
        (avgsA_sub, avgsB_sub, avgsA_nosub, avgsB_nosub, mags_sub, mags_nosub) = single_sweep(t_name,
                                                                        awg,
                                                                        board,
                                                                        num_patterns,
                                                                        params,
                                                                        extra_column = [param2, new_param])

        #avgsA has shape [num_patterns][num_sweeps]
        #finalA should have shape [num_patterns][sweepl1][sweepl2]
        f_A_nosub[:, :, sweep_num] = avgsA_nosub
        f_B_nosub[:, :, sweep_num] = avgsB_nosub
        f_A_sub[:, :, sweep_num] = avgsA_sub
        f_B_sub[:, :, sweep_num] = avgsB_sub
        f_M_sub[:, :, sweep_num] = mags_sub
        f_M_nosub[:, :, sweep_num] = mags_nosub
        
        sweep_num += 1

    
    return f_A_nosub, f_B_nosub, f_A_sub, f_B_sub, f_M_sub, f_M_nosub
    
    
