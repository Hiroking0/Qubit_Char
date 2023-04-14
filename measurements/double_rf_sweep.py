# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:38:30 2022

@author: lqc
"""
import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from lib import wave_construction as be
from lib import run_funcs

from instruments.alazar import ATS9870_NPT as npt
from instruments.alazar import atsapi as ats

import tkinter.filedialog as tkf
import yaml

if __name__ == "__main__":
    
    f = open('general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    
    directory = tkf.askdirectory()
    name = directory + "/" + params['name']

    #start_time is the time between triggering the AWG and start of the qubit pulse
    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    wait_time = zero_length * zero_multiple
    
    readout_start = params['readout_start']
    readout = params['readout_duration']
    pattern_repeat = params['pattern_repeat']
    seq_repeat = params['seq_repeat']
    num_patterns = params['num_patterns']
    
    acq_multiples = params['acq_multiples']
    samples_per_ac = 256*acq_multiples #length of acquisition in nS must be n*256
    
    GHz=1e9
    MHz=1e6
    kHz=1e3

    p1 = params['p1']
    p2 = params['p2']

    if p1 == 'wq' or p1 == 'wr':
        p1start = params['p1start']*GHz
        p1stop = params['p1stop']*GHz
        p1step = params['p1step']*GHz
    else:
        p1start = params['p1start']
        p1stop = params['p1stop']
        p1step = params['p1step']

    if p2 == 'wq' or p2 == 'wr':
        p2start = params['p2start']*GHz
        p2stop = params['p2stop']*GHz
        p2step = params['p2step']*GHz 
    else:
        p2start = params['p2start']
        p2stop = params['p2stop']
        p2step = params['p2step']
        

    #sweep power J7201B
    avg_start = params['avg_start']
    avg_length = params['avg_length']

    wlen = readout_start + readout + wait_time
    
    awg = be.get_awg()
    
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, wlen)
    run_funcs.init_params(params)

    for i in range(num_patterns):
        awg.set_seq_element_loop_cnt(i+1, pattern_repeat)

    board = ats.Board(systemId = 1, boardId = 1)
    npt.ConfigureBoard(board)

    #returns arrays of channel A and B, averages of averages
    #shape is [num_patterns][p1 sweep length][p2 sweep length]
    cAp, cBp = run_funcs.double_sweep(name, awg, board, p1, p1start, p1stop, p1step, p2, p2start, p2stop, p2step, samples_per_ac, num_patterns, pattern_repeat, seq_repeat, wlen, avg_start, avg_length)

    plt.figure()
    
    y = np.arange(p1start, p1stop, p1step)
    x = np.arange(p2start, p2stop, p2step)
    
    plt.subplot(2,2,1)
    plt.pcolormesh(x, y, cAp[0])
    plt.xlabel("readout frequency (GHz)")
    plt.ylabel("readout attenuation (db)")
    plt.title('channel A')
    
    plt.subplot(2,2,2)
    plt.pcolormesh(x, y, cBp[0])
    plt.xlabel("readout frequency (GHz)")
    plt.ylabel("readout attenuation (db)")
    plt.title('channel B')

    
    mag_arr = np.zeros(np.shape(cAp[0]))
    for i in range(len(cAp[0])):
        for j in range(len(cAp[0][0])):
            mag_arr[i][j] = np.sqrt(cAp[0][i][j]**2 + cBp[0][i][j]**2)
    
    plt.subplot(2,2,3)
    plt.pcolormesh(x, y, mag_arr)
    plt.xlabel("readout frequency (GHz)")
    plt.ylabel("readout attenuation (db)")
    plt.title('Magnitude')
    
    plt.show()
 
'''data=np.stack([vsource, volts, currs], axis=1)
now = datetime.now()
Date = now.strftime("%Y%m%d_%H%M%S")

root = tk.Tk()
pathName = askdirectory(title='Select Folder')
root.withdraw()

print('Data file path: ', pathName + '/' + fileName + '_' + Date)
 
 # Save data to csv file
file = "".join((pathName, '\\', fileName, '_', Date, '.txt'))
with open(file, 'w', newline='') as output:
     header = ['Applied Voltage', ' Measured Voltage', ' Measured Current']
     wr = csv.writer(output, delimiter=',', quoting=csv.QUOTE_NONE)
     wr.writerow(header)
     for i in data:
         # format data (0.00e+00)
         
         wr.writerow([str(x) for x in i])
     output.write('\n')
print('Done ' + Date)'''
