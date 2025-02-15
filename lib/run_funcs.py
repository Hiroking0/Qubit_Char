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
#from instruments.TekAwg import tek_awg as tawg
from multiprocessing import Process, Manager, Queue
import time
import numpy as np
from threading import Thread
from multiprocessing import Process
import pyvisa as visa
import csv
import matplotlib.pyplot as plt
import queue
import pickle as pkl

class Data_Arrs:
    """
    A class for managing data arrays.

    Parameters:
        *args: Data arrays (a_nosub, a_sub, b_nosub, b_sub, mags_nosub, mags_sub, readout_A, readout_B).

    Attributes:
        a_nosub (numpy.ndarray): Data array A without subtraction.
        a_sub (numpy.ndarray): Data array A with subtraction.
        b_nosub (numpy.ndarray): Data array B without subtraction.
        b_sub (numpy.ndarray): Data array B with subtraction.
        mags_nosub (numpy.ndarray): Magnitudes without subtraction.
        mags_sub (numpy.ndarray): Magnitudes with subtraction.
        readout_A (numpy.ndarray): Readout A data.
        readout_B (numpy.ndarray): Readout B data.
    """

    def __init__(self, *args):
        self.a_nosub = args[0]
        self.a_sub = args[1]
        self.b_nosub = args[2]
        self.b_sub = args[3]
        self.mags_nosub = args[4]
        self.mags_sub = args[5]
        self.readout_A = args[6]
        self.readout_B = args[7]

    def save(self, name):
        """
        Save the data arrays to a pickle file.

        Parameters:
            name (str): The name of the pickle file.
        """
        with open(name + '.pkl', 'wb') as pickle_file:
            pkl.dump(self, pickle_file)

    def get_data_arrs(self):
        """
        Get the data arrays.

        Returns:
            tuple: A tuple containing data arrays (a_nosub, a_sub, b_nosub, b_sub, mags_nosub, mags_sub, readout_A, readout_B).
        """
        return (self.a_nosub, self.a_sub, self.b_nosub, self.b_sub, self.mags_nosub, self.mags_sub, self.readout_A, self.readout_B)

    def get_avgs(self, theta=0):
        """
        Calculate average values.

        Parameters:
            theta (float, optional): Rotation angle (default is 0).

        Returns:
            tuple: A tuple containing arrays (new_pattern_avgs_cA, new_pattern_avgs_cA_sub, new_pattern_avgs_cB, new_pattern_avgs_cB_sub, mags, mags_sub).
        """
        length = len(self.a_nosub)
        
        # Initialize arrays to store average values.
        pattern_avgs_cA = np.zeros(length)
        pattern_avgs_cB = np.zeros(length)
        mags = np.zeros(length)
        
        pattern_avgs_cA_sub = np.zeros(length)
        pattern_avgs_cB_sub = np.zeros(length)
        mags_sub = np.zeros(length)
        
        # Calculate average values for each data array.
        for i in range(len(mags)):
            pattern_avgs_cA[i] = np.average(self.a_nosub[i])
            pattern_avgs_cB[i] = np.average(self.b_nosub[i])
            mags[i] = np.average(self.mags_nosub[i])
            
            pattern_avgs_cA_sub[i] = np.average(self.a_sub[i])
            pattern_avgs_cB_sub[i] = np.average(self.b_sub[i])
            mags_sub[i] = np.average(self.mags_sub[i])

        # Initialize arrays to store complex values and angles.
        complex_arr = np.zeros(length, dtype=np.complex_)
        complex_arr_sub = np.zeros(length, dtype=np.complex_)
        angle_arr = np.angle(complex_arr_sub.flatten())
        
        # Calculate rotation angle in radians.
        theta = np.radians(theta)
        exp = np.exp(1j*theta) # Rotation
        
        # Apply the rotation to all data points.
        for j in range(length):
            t_i = pattern_avgs_cA[j]
            t_q = pattern_avgs_cB[j]
            t_new = np.multiply(t_i+1j*t_q, exp)
            complex_arr[j] = t_new

            t_i_sub = pattern_avgs_cA_sub[j]
            t_q_sub = pattern_avgs_cB_sub[j]
            t_new_sub = np.multiply(t_i_sub+1j*t_q_sub, exp)
            complex_arr_sub[j] = t_new_sub
        
        # Extract the real and imaginary part of the complex array.
        new_pattern_avgs_cA = np.real(complex_arr)
        new_pattern_avgs_cB = np.imag(complex_arr)
        new_pattern_avgs_cA_sub = np.real(complex_arr_sub)
        new_pattern_avgs_cB_sub = np.imag(complex_arr_sub)

        return (new_pattern_avgs_cA, new_pattern_avgs_cA_sub, new_pattern_avgs_cB, new_pattern_avgs_cB_sub, mags, mags_sub)

def initialize_awg(awg, num_patterns, pattern_repeat, decimation):
    """
    Initialize the AWG with specific parameters.

    Parameters:
        awg (object): The AWG object.
        num_patterns (int): Number of patterns.
        pattern_repeat (int): Number of pattern repeats.
        decimation (float): Decimation value.
    """
    awg.set_chan_state(1, [1,2,3,4])    
    
    for i in range(1, num_patterns):
        awg.set_seq_element_goto_state(i, 0)
        awg.set_seq_element_loop_cnt(i, pattern_repeat)
    awg.set_seq_element_goto_state(num_patterns, 1)
    
    #set sampling rate
    new_freq = 1/decimation
    awg.set_freq(str(new_freq)+"GHZ")

def init_params(params):
    """
    Initialize instrument parameters.

    Parameters:
        params (dict): A dictionary of parameter values.
    """
    rm = visa.ResourceManager()
    q_rf = RF.RF_source(rm, "TCPIP0::172.20.1.7::5025::SOCKET")
    r_rf = RF.RF_source(rm, "TCPIP0::172.20.1.8::5025::SOCKET")
    qef_rf = RF.RF_source(rm, "TCPIP0::172.20.1.17::5025::SOCKET")
    q_att = ATT.Atten(rm, "TCPIP0::172.20.1.6::5025::SOCKET")
    r_att = ATT.Atten(rm, "TCPIP0::172.20.1.9::5025::SOCKET")
    twpa_rf = RF.RF_source(rm, "TCPIP0::172.20.1.11::5025::SOCKET")
    q_rf.set_freq(params['set_wq'])
    r_rf.set_freq(params['set_wr'])
    qef_rf.set_freq(params['set_wef'])
    q_rf.set_power(params['set_pq'])
    r_rf.set_power(params['set_pr'])
    qef_rf.set_power(params['set_pwef'])
    q_att.set_attenuation(params['set_q_att'])
    r_att.set_attenuation(params['set_r_att'])
    twpa_rf.set_power(params['set_p_twpa'])
    twpa_rf.set_freq(params['set_w_twpa'])

def turn_on_3rf():
    """
    Turn on three RF sources.
    """
    rm = visa.ResourceManager()
    q_rf = RF.RF_source(rm, "TCPIP0::172.20.1.7::5025::SOCKET")
    r_rf = RF.RF_source(rm, "TCPIP0::172.20.1.8::5025::SOCKET")
    qef_rf = RF.RF_source(rm, "TCPIP0::172.20.1.17::5025::SOCKET")
    q_rf.enable_out()
    r_rf.enable_out()
    qef_rf.enable_out()

def turn_on_2rf():
    """
    Turn on two RF sources.
    """
    rm = visa.ResourceManager()
    q_rf = RF.RF_source(rm, "TCPIP0::172.20.1.7::5025::SOCKET")
    r_rf = RF.RF_source(rm, "TCPIP0::172.20.1.8::5025::SOCKET")
    q_rf.enable_out()
    r_rf.enable_out()

# Turns on the RF for wr
def turn_on_rf():
    rm = visa.ResourceManager()
    r_rf = RF.RF_source(rm, "TCPIP0::172.20.1.8::5025::SOCKET")
    r_rf.enable_out()

def turn_on_rf_wq():
    rm = visa.ResourceManager()
    q_rf = RF.RF_source(rm, "TCPIP0::172.20.1.7::5025::SOCKET")
    q_rf.enable_out()

# Turns off the RF
def turn_off_inst():
    """
    Turn off RF sources.
    """
    rm = visa.ResourceManager()
    q_rf = RF.RF_source(rm, "TCPIP0::172.20.1.7::5025::SOCKET")
    r_rf = RF.RF_source(rm, "TCPIP0::172.20.1.8::5025::SOCKET")
    qef_rf = RF.RF_source(rm, "TCPIP0::172.20.1.17::5025::SOCKET")
    q_rf.disable_out()
    r_rf.disable_out()
    qef_rf.disable_out()
    
def run_and_acquire(awg, board, params, num_patterns, path,live_plot = False) -> Data_Arrs:
    """
    Run a sequence on the AWG and acquire data.

    Parameters:
        awg (object): The AWG object.
        board (object): The board object.
        params (dict): A dictionary of parameter values.
        num_patterns (int): Number of patterns.
        path (str): The path to save data.

    Returns:
        Data_Arrs: An instance of Data_Arrs containing acquired data.
    """
    save_raw = False
    manager = Manager()
    manager1 = Manager()
    que = manager.Queue()
    data_queue = manager1.Queue()
    #acproc = Thread(target = lambda q, board, params, num_patterns, path, raw, live:
    #                        q.put(npt.AcquireData(board, params, num_patterns, path, raw, live)), 
    #                        args = (que, board, params, num_patterns, path, save_raw, live_plot))

    #acproc = Process(target = lambda q, params, num_patterns, path, raw, live:
    #                        q.put(npt.AcquireData(params, num_patterns, path, raw, live)), 
    #                        args = (que, params, num_patterns, path, save_raw, live_plot))

    que.put((params, num_patterns, path, save_raw, live_plot))

    acproc = Process(target = npt.AcquireData, args = (que,data_queue,))
    #plotting_process = Process(target = npt.live_plot, args = (params, num_patterns,que,data_queue,))
    
    acproc.start()
    #plotting_process.start()
    time.sleep(.5)
    awg.run()
    while not data_queue.empty():
        data_queue.get()
    acproc.join()
    #plotting_process.join()
    arrs = que.get()
    awg.stop()
    data_arrs = Data_Arrs(*arrs)
    return data_arrs

    
    
#this function will take one of the do_all functions as a parameter
#sweep param should be 'wq', 'pq', 'wr', 'pr', 'att'
#w is for frequency, p for power
#q for qubit, r for readout
#other instrument(s) will not be changed
    
def get_func_call(rm, sweep_param, awg):
    """
    Get a function call for a specific instrument parameter.

    Parameters:
        rm (object): The visa.ResourceManager object.
        sweep_param (str): The parameter to sweep.
        awg (object): The AWG object.

    Returns:
        function: The function to call to set the parameter.
    """
    qubit_addr = "TCPIP0::172.20.1.7::5025::SOCKET"
    readout_addr = "TCPIP0::172.20.1.17::5025::SOCKET"
    q_atten_addr = "TCPIP0::172.20.1.6::5025::SOCKET"
    r_atten_addr = "TCPIP0::172.20.1.9::5025::SOCKET"
    twpa_addr = "TCPIP0::172.20.1.11::5025::SOCKET"
    ef_qubit_addr = 'TCPIP0::172.20.1.17::5025::SOCKET'
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
        'wef': ef_qubit_addr,
        'pef':ef_qubit_addr
        #'amp':amp_addr
    }
    
    inst_table = {
        'wr': RF.RF_source,
        'wq': RF.RF_source,
        'pq': RF.RF_source,
        'pr': RF.RF_source,
        'wef': RF.RF_source,
        'pef': RF.RF_source,
        'r_att': ATT.Atten,
        'q_att': ATT.Atten,
        'p_twpa': RF.RF_source,
        'w_twpa': RF.RF_source
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
        'p_twpa': 'set_power',
        'w_twpa': 'set_freq',
        'wef': 'set_freq',
        'pef': 'set_power',
        'amp' : "set_amplitude"
        }
    
    func_call = getattr(inst, func_table[sweep_param])
    return func_call


#extra_column used by double sweep function to store second parameter.
#It should be python list of ["parameter name", value]
def single_sweep2(name,
                 awg,
                 board,
                 num_patterns,
                 params,
                 extra_column = None,
                 live_plot = False):
    #1.8 rf is for qubit
    #1.7 rf is for readout
    
    sweep_param = params['p2']
    start = params['p2start']
    stop = params['p2stop']
    step = params['p2step']
    
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
        
        data = run_and_acquire(awg,
                                board,
                                params,
                                num_patterns,
                                path = name)
        #(pattern_avgs_cA, pattern_avgs_cB, mags, pattern_avgs_cA_sub, pattern_avgs_cB_sub, mags_sub)
        #avgsA should be array of shape(num_patterns, sweep_num, x)
        #(pattern_avgs_cA, pattern_avgs_cA_sub, pattern_avgs_cB, pattern_avgs_cB_sub, mags, mags_sub)
        (t_an, t_as, t_bn, t_bs, m_ns, m_s) = data.get_avgs()



        
        avgsA_sub[:, sweep_num] = t_as
        avgsB_sub[:, sweep_num] = t_bs
        avgsA_nosub[:, sweep_num] = t_an
        avgsB_nosub[:, sweep_num] = t_bn
        mags_sub[:, sweep_num] = m_s
        mags_nosub[:, sweep_num] = m_ns
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

    #f_name = f"{sweep_param}_{date}.csv"
    f_name = f"{sweep_param}.csv"
    with open(f"{name}_{f_name}", 'w', newline='') as output:
    #with open(name + '_' + f_name, 'w', newline='') as output:
        wr = csv.writer(output, delimiter=',', quoting=csv.QUOTE_NONE)
        if extra_column != None:
            header = header=['chA_sub2','chB_sub2','mag_sub2', 'chA_nosub2', 'chB_nosub2', 'mag_nosub2', 'pattern_num2', str(sweep_param), extra_column[0]]
        else:
            header=['chA_sub2','chB_sub2','mag_sub2', 'chA_nosub2', 'chB_nosub2', 'mag_nosub2', 'pattern_num2', str(sweep_param)]
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
    
def single_sweep(name, awg, board, num_patterns, params, extra_column=None, live_plot=False):
    """
    Perform a single parameter sweep.

    Parameters:
        name (str): The name of the data file.
        awg (object): The AWG object.
        board (object): The board object.
        num_patterns (int): Number of patterns.
        params (dict): A dictionary of parameter values.
        extra_column (list, optional): Extra column information (default is None).
        live_plot (bool, optional): Enable live plotting (default is False).

    Returns:
        tuple: A tuple containing arrays (avgsA_sub, avgsB_sub, avgsA_nosub, avgsB_nosub, mags_sub, mags_nosub).
    """
    #1.8 rf is for qubit
    #1.7 rf is for readout
    

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
        
        data = run_and_acquire(awg,
                                board,
                                params,
                                num_patterns,
                                path = name)
        #(pattern_avgs_cA, pattern_avgs_cB, mags, pattern_avgs_cA_sub, pattern_avgs_cB_sub, mags_sub)
        #avgsA should be array of shape(num_patterns, sweep_num, x)
        #(pattern_avgs_cA, pattern_avgs_cA_sub, pattern_avgs_cB, pattern_avgs_cB_sub, mags, mags_sub)
        (t_an, t_as, t_bn, t_bs, m_ns, m_s) = data.get_avgs()



        
        avgsA_sub[:, sweep_num] = t_as
        avgsB_sub[:, sweep_num] = t_bs
        avgsA_nosub[:, sweep_num] = t_an
        avgsB_nosub[:, sweep_num] = t_bn
        mags_sub[:, sweep_num] = m_s
        mags_nosub[:, sweep_num] = m_ns
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

    #f_name = f"{sweep_param}_{date}.csv"
    f_name = f"{sweep_param}.csv"
    with open(f"{name}_{f_name}", 'w', newline='') as output:
    #with open(name + '_' + f_name, 'w', newline='') as output:
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


def double_sweep(name, awg, board, params, num_patterns, live_plot=False):
    """
    Perform a double parameter sweep.

    Parameters:
        name (str): The name of the data file.
        awg (object): The AWG object.
        board (object): The board object.
        params (dict): A dictionary of parameter values.
        num_patterns (int): Number of patterns.
        live_plot (bool, optional): Enable live plotting (default is False).

    Returns:
        tuple: A tuple containing arrays (f_A_nosub, f_B_nosub, f_A_sub, f_B_sub, f_M_sub, f_M_nosub).
    """
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


