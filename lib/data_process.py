# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:37:25 2022

@author: lqc
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def get_population_v_pattern(arr, thresh, flipped = False):
    plt_arr = np.zeros(len(arr))
    for i in range(len(arr)):
        pe = 0
        #for each pattern, look at every point and see if its above thresh
        for j in arr[i]:
            if (j > thresh and not flipped) or (j < thresh and flipped):
                pe += 1
        pe /= len(arr[0])
        plt_arr[i] = pe

    return plt_arr

def frombin(tot_samples, numAcquisitions, channels = 2, name = 'data.bin'):
    #assume 1 record per buffer
    
    arr = np.fromfile(name, np.ubyte)
    
    #arr is shaped like [A1 for samples, B1 for samples, A2 for samples, B2 for samples]
    
    chA = np.zeros((numAcquisitions, tot_samples))
    chB = np.zeros((numAcquisitions, tot_samples))
    
    if channels == 1:
        for i in range(0, numAcquisitions*tot_samples, tot_samples):
            chA[int(i/tot_samples)] = arr[i:i+tot_samples]
        return chA
    
    else:
        arr = np.reshape(arr, (-1,tot_samples))
        chA = arr[::2]
        chB = arr[1::2]
        #print("cha shape", np.shape(chA))
        return (chA, chB)
    
    
    #plt.plot(arr)
    #plt.show()
    
def fromnump():
    with open('datacA.npy', 'rb') as f:
        chA = np.load(f)
    with open('datacB.npy', 'rb') as f:
        chB = np.load(f)
        
    print(np.shape(chA))
    print(np.shape(chB))

    for i in range(1,len(chA)):
        plt.plot(chA[i])
        #plt.plot(chB[i])
        plt.show()
 

#this function is for averaging each index in parallel. It returns 1 array of
#the average of all index is
def average_all_iterations(arr):
    #print(np.shape(arr))
    l_iter = len(arr[0])
    final_arr = np.zeros(l_iter)
    num_pat = len(arr)
    for i in range(l_iter):
        tsum = np.sum(arr[:,i])/num_pat
        final_arr[i] = tsum
    #print("end of avg all iter shape: ", np.shape(final_arr))
    return final_arr

#this function takes the average from each readout and returns an array of averages

def get_avgs(arr, avg_start, avg_length, subtract = True):
    #print(np.shape(arr))
    final_arr = np.array([])
    for subA in arr:
        tavg = np.average(subA[avg_start : avg_start+avg_length])
        if subtract:
            small = np.average(subA[200:800])
            tavg -= small

        final_arr = np.append(final_arr, tavg)
            
    return final_arr
    
#returns array with each entry being arrays for that pattern i.e final_arr[0] is pattern 1, final_arr[1] is pattern 2 etc
#[pattern #][reptition #][sample index]
def get_p1_p2(arr, num_patterns, pattern_reps, seq_reps):
    #this extracts patterns from the chA/B array
    rec_len = len(arr[0])
    
    arr = np.reshape(arr, (num_patterns*seq_reps, pattern_reps, -1))
    final_arr = np.zeros((num_patterns, pattern_reps*seq_reps, rec_len))

    for i in range(num_patterns):
        tarr = arr[i::num_patterns, :]
        tarr = np.reshape(tarr, (-1, rec_len))
        final_arr[i] = np.squeeze(tarr)
        
    return final_arr

def parse_np_file(arr, num_patterns, pattern_reps, seq_reps, measurement = None):
    arr = np.reshape(arr, (num_patterns * seq_reps, pattern_reps))
    final_arr = np.zeros((num_patterns, pattern_reps*seq_reps))
    for i in range(num_patterns):
        tarr = arr[i::num_patterns]
        tarr = np.reshape(tarr, (-1))
        final_arr[i] = np.squeeze(tarr)
    return final_arr

def plot_np_file(num_patterns, pattern_reps, seq_reps, time_step, path):
    chA_sub = np.load(path + "chA_sub.npy")
    chB_sub = np.load(path + "chB_sub.npy")
    chA_nosub = np.load(path + "chA_nosub.npy")
    chB_nosub = np.load(path + "chB_nosub.npy")
    
    
    #chA = parse_np_file(chA, num_patterns, pattern_reps, seq_reps)
    #hB = parse_np_file(chB, num_patterns, pattern_reps, seq_reps)
    num_bins = 500

    
    plt.subplot(2,2,1)
    plt.hist(chA_nosub.flatten(), bins = num_bins, histtype = 'step')
    plt.title("chA no subtraction")
    
    plt.subplot(2,2,2)
    plt.hist(chB_nosub.flatten(), bins = num_bins, histtype = 'step')
    plt.title("chB no subtraction")
    
    
    plt.subplot(2,2,3)
    plt.hist(chA_sub.flatten(), bins = num_bins, histtype = 'step')
    plt.title("chA with subtraction")
    
    plt.subplot(2,2,4)
    plt.hist(chB_sub.flatten(), bins = num_bins, histtype = 'step')
    plt.title("chB with subtraction")
    
    plt.figure()

    pattern_avgs_cA = np.zeros(num_patterns)
    pattern_avgs_cB = np.zeros(num_patterns)
    pattern_avgs_cA_sub = np.zeros(num_patterns)
    pattern_avgs_cB_sub = np.zeros(num_patterns)
    mags = np.zeros(num_patterns)
    mags_sub = np.zeros(num_patterns)
    #print(np.shape(chA_nosub))
    for i in range(num_patterns):
        pattern_avgs_cA[i] = np.average(chA_nosub[i])
        pattern_avgs_cB[i] = np.average(chB_nosub[i])
        mags[i] = np.average(np.sqrt(chB_nosub[i] ** 2 + chA_nosub[i] ** 2))
        
        pattern_avgs_cA_sub[i] = np.average(chA_sub[i])
        pattern_avgs_cB_sub[i] = np.average(chB_sub[i])
        mags_sub[i] = np.average(np.sqrt(chB_sub[i] ** 2 + chA_sub[i] ** 2))
        
    x = [i*time_step for i in range(num_patterns)]
    font_size = 10
    #x = np.arange(num_patterns)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    
    plt.subplot(2,3,1)
    plt.plot(x, pattern_avgs_cA)
    plt.title("chA nosub", fontsize=font_size)
    plt.xlabel('time (ns)', fontsize=font_size)
    plt.ylabel('Voltage (V)', fontsize=font_size)
    
    print(pattern_avgs_cB)
    plt.subplot(2,3,2)
    plt.plot(x, pattern_avgs_cB)
    plt.title("chB nosub", fontsize=font_size)
    plt.xlabel('time (ns)', fontsize=font_size)
    plt.ylabel('Voltage (V)', fontsize=font_size)
    
    plt.subplot(2,3,3)
    plt.plot(x, mags)
    plt.title("mag nosub", fontsize=font_size)
    plt.xlabel('time (ns)', fontsize=font_size)
    plt.ylabel('Voltage (V)', fontsize=font_size)
    
    plt.subplot(2,3,4)
    plt.plot(x, pattern_avgs_cA_sub)
    plt.title("chA sub", fontsize=font_size)
    plt.xlabel('time (ns)', fontsize=font_size)
    plt.ylabel('Voltage (V)', fontsize=font_size)
    
    plt.subplot(2,3,5)
    plt.plot(x, pattern_avgs_cB_sub)
    
    plt.title("chB sub", fontsize=font_size)
    plt.xlabel('time (ns)', fontsize=font_size)
    plt.ylabel('Voltage (V)', fontsize=font_size)
    
    plt.subplot(2,3,6)
    plt.plot(x, mags_sub)
    plt.title("mag sub", fontsize=font_size)
    plt.xlabel('time (ns)', fontsize=font_size)
    plt.ylabel('Voltage (V)', fontsize=font_size)
    
    plt.get_current_fig_manager().window.showMaximized()
    #plt.tight_layout()
    plt.savefig(path + "_pic", dpi= 300, pad_inches = 0, bbox_inches = 'tight')
    plt.show()


def plot_colors(cAp1, cBp1):
    ind = 0
    for i in range(0, len(cAp1), pattern_reps):
        plt.subplot(seq_reps, 4, (4*ind)+1)
        plt.pcolormesh(cAp1[i:i+pattern_reps])
        
        plt.subplot(seq_reps, 4, (4*ind)+2)
        #plt.pcolormesh(cAp2[i:i+pattern_reps])
        
        plt.subplot(seq_reps, 4, (4*ind)+3)
        plt.pcolormesh(cBp1[i:i+pattern_reps])
        
        plt.subplot(seq_reps, 4, (4*ind)+4)
        #plt.pcolormesh(cBp2[i:i+pattern_reps])
        ind += 1
    plt.show()

def plot_average_iterations(cAp1, cBp1):
    #overlaying averages here
    plt.subplot(2,1,1)
    plt.plot(average_all_iterations(cAp1))
    #plt.plot(average_all_iterations(cAp2))
    plt.title('channel A average p1 p2')
    
    plt.subplot(2,1,2)
    plt.plot(average_all_iterations(cBp1))
    #plt.plot(average_all_iterations(cBp2))
    plt.title('channel B average p1 p2')
    
def plot_iq(cAp, cBp):
    #iq here
    cAp1av = get_avgs(cAp)
    #cAp2av = get_avgs(cAp2)

    cBp1av = get_avgs(cBp)
    #cBp2av = get_avgs(cBp2)

    plt.scatter(cAp1av, cBp1av)
    #plt.scatter(cAp2av, cBp2av)
    plt.legend(['pattern 1', 'pattern 2'])
    plt.show()

def plot_histogram(arr):
    #histogram here
    num_patterns = len(arr)
    
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.5)
    for i in range(num_patterns):
        plt.hist(arr[i], bins = 100, alpha = .7)
    #plt.title("chA no subtraction")
    #cAp1av = get_avgs(cAp)
    #cBp1av = get_avgs(cBp)
    plt.legend(["no pulse", "pulse"], loc = 'upper center')
    plt.xlabel("V")
    plt.ylabel("count")
    plt.show()



def plot_all(chA, chB, num_patterns, pattern_reps, seq_reps, avg_start, avg_length, large_data_plot = False):
    bins = 150
    cAp = get_p1_p2(chA, num_patterns, pattern_reps, seq_reps)
    cBp = get_p1_p2(chB, num_patterns, pattern_reps, seq_reps)
    
    #things to plot
    #heat plot for each seq_rep
    #overlaying of averaging of cAo1 p2 etc...
    #ChA vs ChB
    
    #so, I think there should be two plots. One for the heat maps, one for everything else
    
    #heat map here
    if large_data_plot:
        ind = 0
        #print(np.shape(cAp1))
        for i in range(0, len(cAp1), pattern_reps):
            plt.subplot(seq_reps, 4, (4*ind)+1)
            plt.pcolormesh(cAp1[i:i+pattern_reps])
            
            plt.subplot(seq_reps, 4, (4*ind)+3)
            plt.pcolormesh(cBp1[i:i+pattern_reps])
            
            if num_patterns == 2:
                plt.subplot(seq_reps, 4, (4*ind)+2)
                plt.pcolormesh(cAp2[i:i+pattern_reps])
                
                plt.subplot(seq_reps, 4, (4*ind)+4)
                plt.pcolormesh(cBp2[i:i+pattern_reps])
            ind += 1
        
        plt.figure()
    
    #overlaying averages here
    plt.subplot(3,3,1)
    ts = time.time()
    for i in range(num_patterns):
        plt.plot(average_all_iterations(cAp[i]))
    
    te = time.time()
    plt.title('channel A average p1 p2')
    #plt.legend(["pattern "+str(i) for i in range(num_patterns)])
    
    #averages for channel B
    plt.subplot(3,3,2)
    for i in range(num_patterns):
        plt.plot(average_all_iterations(cBp[i]))
    plt.title('channel B average p1 p2')
    #plt.legend(["pattern "+str(i) for i in range(num_patterns)])
    
    #iq here with subtraction
    plt.subplot(3,3,3)
    for i in range(num_patterns):
        cApn_av = get_avgs(cAp[i], avg_start, avg_length, False)
        cBpn_av = get_avgs(cBp[i], avg_start, avg_length, False)
        plt.scatter(cApn_av, cBpn_av, alpha = .3)

    #plt.legend(["pattern "+str(i) for i in range(num_patterns)])
    plt.title("I vs Q without subtraction")
    #plt.legend(['pattern 1', 'pattern 2'])
    
    #histogram here
    
    plt.subplot(3,3,4)
    s = time.time()
    for i in range(num_patterns):
        cApn_av = get_avgs(cAp[i], avg_start, avg_length)
        #cBpn_av = get_avgs(cBp[i], avg_start, avg_length)
        #mags = np.zeros(len(cApn_av))
        
        #for j in range(len(cApn_av)):
        #    t_mag = np.sqrt(cApn_av[j] ** 2 + cBpn_av[j] ** 2)
        #    mags[j] = t_mag
            
        plt.hist(cApn_av, bins = bins, histtype = 'step')
        
        
    plt.title('chA with subtraction')
    plt.legend(["pattern "+str(i) for i in range(num_patterns)])
    
    
    plt.subplot(3,3,5)
    for i in range(num_patterns):
        #cApn_av = get_avgs(cAp[i], avg_start, avg_length)
        cBpn_av = get_avgs(cBp[i], avg_start, avg_length)
        #mags = np.zeros(len(cApn_av))
        
        #for j in range(len(cApn_av)):
        #    t_mag = np.sqrt(cApn_av[j] ** 2 + cBpn_av[j] ** 2)
        #    mags[j] = t_mag
            
        plt.hist(cBpn_av, bins = bins, histtype = 'step')
        
        
    plt.title('chB with subtraction')
    plt.legend(["pattern "+str(i) for i in range(num_patterns)])
    #plt.legend(["pattern "+str(i) for i in range(num_patterns)])

    
    plt.subplot(3,3,6)
    for i in range(num_patterns):
        cApn_av = get_avgs(cAp[i], avg_start, avg_length, subtract = True)
        cBpn_av = get_avgs(cBp[i], avg_start, avg_length, subtract = True)
        mags = np.zeros(len(cApn_av))
        for j in range(len(cApn_av)):
            t_mag = np.sqrt(cApn_av[j] ** 2 + cBpn_av[j] ** 2)
            mags[j] = t_mag
        plt.hist(mags, bins = bins, histtype = 'step')
        plt.title("Magnitude with subtraction")
    
    plt.subplot(3,3,7)
    for i in range(num_patterns):
        cApn_av = get_avgs(cAp[i], avg_start, avg_length, subtract = False)
        plt.hist(cApn_av, bins = bins, histtype = 'step')
    plt.title("chA no subtraction")
    
    plt.subplot(3,3,8)
    for i in range(num_patterns):
        cBpn_av = get_avgs(cBp[i], avg_start, avg_length, subtract = False)
        plt.hist(cBpn_av, bins = bins, histtype = 'step')
    #plt.legend(["pattern "+str(i) for i in range(num_patterns)])
    plt.title("chB no subtraction")
    
    plt.subplot(3,3,9)
    for i in range(num_patterns):
        cApn_av = get_avgs(cAp[i], avg_start, avg_length, subtract = False)
        cBpn_av = get_avgs(cBp[i], avg_start, avg_length, subtract = False)
        mags = np.zeros(len(cApn_av))
        for j in range(len(cApn_av)):
            t_mag = np.sqrt(cApn_av[j] ** 2 + cBpn_av[j] ** 2)
            mags[j] = t_mag
        plt.hist(mags, bins = bins, histtype = 'step')
        plt.title("Magnitude no subtraction")

        #plt.hist(np.imag(complex_arr), bins = 100)

    plt.show()


if __name__ == "__main__":
    num_patterns = 2
    pattern_reps = 10
    seq_reps = 3
    (chA, chB) = frombin(numAcquisitions = num_patterns*pattern_reps*seq_reps)
    plot_all(chA, chB, num_patterns, pattern_reps, seq_reps)
