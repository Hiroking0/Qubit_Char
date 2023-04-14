# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:00:50 2021

@author: Crow108
"""

import numpy as np
#import daq_programs
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from scipy.optimize import curve_fit
import yaml

def objective_T1(x, a, b, c):
	return a + (b*np.exp(-x/c))

def fit_T1(y, init_a, init_b, init_c, num_points, max_length):
    x_data = np.linspace(0, max_length, num_points)
    y_data = y
    initial  = [init_a,init_b,init_c]
    popt, _ = curve_fit(objective_T1, x_data, y_data, p0 = initial)
    a, b, c= popt
    
    new_data = objective_T1(x_data,a,b,c)
    
    return new_data, a, b, c


'''
def T1_loop(hrs = 1.0, t = 300 , avg = 800):
    s = 600 # sleep time in s
    runs =  (hrs*60*60)//s; runs = int(runs)
    T1_list = np.zeros((runs)) 
    
    for i in range (runs):
        try:
            rec_avg_all, rec_readout, rec_avg_vs_pats = daq_programs.run_daq2(51, avg, verbose=0)
        except Exception as e:
            if "ApiPllNotLocked" in str(e) : 
                rec_avg_all, rec_readout, rec_avg_vs_pats = daq_programs.run_daq2(51, avg, verbose=0)
        T1_list[i] = fit(rec_avg_vs_pats[1],t)
        time.sleep(s)
        
    plt.plot(np.linspace(0,hrs,runs),T1_list)
    return T1_list
        

'''
if __name__ == "__main__":
    
    f = open('../general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    
    fn = askopenfilename()
    arr = np.load(fn)
    print(np.shape(arr))
    print(arr.dtype.metadata)
    #first get pattern avgs
    avgs = np.zeros(len(arr))
    for i in range(len(arr)):
        avgs[i] = np.average(arr[i])
        
    a = 228
    b = -.3
    c = 10679
    longest_T1 = params['T1_final_gap']
    shortest_T1 = params['T1_init_gap']
    num_patterns = len(arr)
    
    fit_data, new_a, new_b, new_c = fit_T1(avgs, a, b, c, num_patterns, longest_T1)
    print("final data:")
    print("offset: ", new_a)
    print("amplitude: ", new_b)
    print("tau: ", new_c)
    
    x = np.linspace(shortest_T1,longest_T1, num_patterns)
    plt.plot(x, avgs, 'ko')
    plt.plot(x, fit_data, 'r')
    plt.xlabel("$t_{T1}$ (ns)")
    plt.ylabel("V")
    plt.title("T1 Measurement")
    plt.show()
    
    