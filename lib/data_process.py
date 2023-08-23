# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:37:25 2022

@author: lqc
"""
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from .run_funcs import Data_Arrs  # Import a custom module named 'run_funcs' that contains a class 'Data_Arrs'

# Define a function to calculate population vs. pattern
def get_population_v_pattern(arr, thresh, GE=0, flipped=False):
    # Initialize an array to store population values
    plt_arr = np.zeros(len(arr))

    # Loop through patterns in the 'arr'
    for i in range(len(arr)):
        pe = 0
        # Iterate through each point in the pattern
        for j in arr[i]:
            # Check if the point is above 'thresh' (or below if 'flipped' is True)
            if (j > thresh and not flipped) or (j < thresh and flipped):
                pe += 1
        # Calculate population as a ratio of points above 'thresh'
        pe /= len(arr[0])
        plt_arr[i] = pe

    # Return the array of population values
    return plt_arr

# Define a function to calculate effective temperature
def eff_temp(arr, thresh, wq, flipped=False):
    kb = 1.380649e-23  # Boltzmann constant
    h = 6.62607015e-34  # Planck's constant
    del_E = (-h * wq)  # Energy difference

    popG = np.zeros(len(arr))
    popE = np.zeros(len(arr))

    # Calculate populations for the ground state ('popG')
    pe = 0
    pep = 0
    for j in arr[0]:
        if (j > thresh and not flipped) or (j < thresh and flipped):
            pe += 1
        else:
            pep += 1
    pe /= len(arr[0])
    pep /= len(arr[0])
    popG = [pe, pep]

    # Calculate populations for the excited state ('popE')
    pe = 0
    pep = 0
    for j in arr[1]:
        if (j > thresh and not flipped) or (j < thresh and flipped):
            pe += 1
        else:
            pep += 1
    pe /= len(arr[0])
    pep /= len(arr[0])
    popE = [pe, pep]

    # Calculate effective temperature with and without a pulse
    denom = kb * np.log(popG[1] / popG[0])
    T = np.abs(del_E / denom)
    print("Effective temperature (No pulse) (mK):", T * (10 ** 3))

    denom = kb * np.log(popE[1] / popE[0])
    T = np.abs(del_E / denom)
    print("Effective temperature (Pulse) (mK):", T * (10 ** 3))

# Define a function to read data from a binary file
def frombin(tot_samples, numAcquisitions, channels=2, name='data.bin'):
    # Assume 1 record per buffer
    # Read data from the binary file into an array
    arr = np.fromfile(name, np.ubyte)
    # arr is shaped like [A1 for samples, B1 for samples, A2 for samples, B2 for samples]

    # Initialize arrays for channel A and channel B data
    chA = np.zeros((numAcquisitions, tot_samples))
    chB = np.zeros((numAcquisitions, tot_samples))

    if channels == 1:
        # If there's only one channel, separate the data into chA
        for i in range(0, numAcquisitions * tot_samples, tot_samples):
            chA[int(i / tot_samples)] = arr[i:i + tot_samples]
        return chA
    else:
        # If there are two channels, separate the data into chA and chB
        arr = np.reshape(arr, (-1, tot_samples))
        chA = arr[::2]
        chB = arr[1::2]
        return (chA, chB)

# Define a function to read data from NumPy binary files
def fromnump():
    # Load data from 'datacA.npy' and 'datacB.npy' files
    with open('datacA.npy', 'rb') as f:
        chA = np.load(f)
    with open('datacB.npy', 'rb') as f:
        chB = np.load(f)

    print(np.shape(chA))
    print(np.shape(chB))

    for i in range(1, len(chA)):
        plt.plot(chA[i])
        # plt.plot(chB[i])
        plt.show()

# Define a function to average values in parallel for each index
def average_all_iterations(arr):
    l_iter = len(arr[0])
    final_arr = np.zeros(l_iter)
    num_pat = len(arr)
    for i in range(l_iter):
        tsum = np.sum(arr[:, i]) / num_pat
        final_arr[i] = tsum
    return final_arr

# Define a function to calculate averages for each pattern
def get_avgs(arr, avg_start, avg_length, subtract=True):
    final_arr = np.array([])
    for subA in arr:
        tavg = np.average(subA[avg_start:avg_start + avg_length])
        if subtract:
            small = np.average(subA[200:800])
            tavg -= small
        final_arr = np.append(final_arr, tavg)
    return final_arr

# Define a function to organize patterns and repetitions
def get_p1_p2(arr, num_patterns, pattern_reps, seq_reps):
    rec_len = len(arr[0])
    arr = np.reshape(arr, (num_patterns * seq_reps, pattern_reps, -1))
    final_arr = np.zeros((num_patterns, pattern_reps * seq_reps, rec_len))

    for i in range(num_patterns):
        tarr = arr[i::num_patterns, :]
        tarr = np.reshape(tarr, (-1, rec_len))
        final_arr[i] = np.squeeze(tarr)

    return final_arr

# Define a function to rotate data
def rotation(data, angle):
    # Extract data arrays
    (a_nosub, a_sub, b_nosub, b_sub, mags_nosub, mags_sub, readout_A, readout_B) = data.get_data_arrs()

    complex_arr = np.zeros((len(a_nosub), len(a_nosub[0])), dtype=np.complex_)
    complex_arr_sub = np.zeros((len(a_nosub), len(a_nosub[0])), dtype=np.complex_)
    
    # Calculate the rotation angle in radians
    angle_arr = np.angle(complex_arr_sub.flatten())
    theta = np.average(angle_arr)
    theta = np.radians(angle)

    exp = np.exp(1j * theta)

    for i in range(len(a_nosub)):
        for j in range(len(a_nosub[0])):
            t_i = a_nosub[i, j]
            t_q = b_nosub[i, j]
            t_new = np.multiply(t_i + 1j * t_q, exp)
            complex_arr[i, j] = t_new

            t_i_sub = a_sub[i, j]
            t_q_sub = b_sub[i, j]
            t_new_sub = np.multiply(t_i_sub + 1j * t_q_sub, exp)
            complex_arr_sub[i, j] = t_new_sub

    new_a_nosub = np.real(complex_arr)
    new_b_nosub = np.imag(complex_arr)

    new_a_sub = np.real(complex_arr_sub)
    new_b_sub = np.imag(complex_arr_sub)

    setattr(data, 'a_nosub', new_a_nosub)
    setattr(data, 'b_nosub', new_b_nosub)

    setattr(data, 'a_sub', new_a_sub)
    setattr(data, 'b_sub', new_b_sub)
    return data

# Define a function to parse data from a NumPy binary file
def parse_np_file(arr, num_patterns, pattern_reps, seq_reps, measurement=None):
    arr = np.reshape(arr, (num_patterns * seq_reps, pattern_reps))
    final_arr = np.zeros((num_patterns, pattern_reps * seq_reps))
    for i in range(num_patterns):
        tarr = arr[i::num_patterns]
        tarr = np.reshape(tarr, (-1))
        final_arr[i] = np.squeeze(tarr)
    return final_arr

# Define a function to plot histograms
def plot_hist(ax, dat, num_bins, title):
    for pattern in dat:
        ax.hist(pattern, bins=num_bins, histtype='step', alpha=0.8)
    ax.set_title(title)
    

# Define a function to plot I vs. Q
def plot_iq(ax, I, Q, title):
    bins = 100
    H, xedges, yedges = np.histogram2d(I.flatten(), Q.flatten(), bins=(bins, bins))
    H = H.T
    ax.imshow(H, interpolation='nearest', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax.set_title(title)
    '''
    #add check for multiple dimensions
    for i in range(len(I)):
        ax.scatter(I[i], Q[i], alpha = .3)
    ax.set_title(title)
    '''
    
# Define a function to plot subplots
def plot_subaxis(ax, y, title):
    for i in range(len(y)):
        ax.plot(y[i], alpha=0.8)
    ax.set_title(title)

# Define a function to plot patterns vs. voltage
def plot_pattern_vs_volt(ax, x, y, title, font_size):
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel('time (ns)', fontsize=font_size)
    ax.set_ylabel('Voltage (V)', fontsize=font_size)

# Define a function to plot data from NumPy binary files
def plot_np_file(data: Data_Arrs, time_step, path=None):
    # Extract data arrays
    (chA_nosub, chA_sub, chB_nosub, chB_sub, mags_nosub, mags_sub, readout_a, readout_b) = data.get_data_arrs()
    num_patterns = 1 if np.ndim(chA_nosub) == 1 else len(chA_nosub)
    num_bins = 200

    # Create subplots for various data representations
    fig, ax_array = plt.subplots(3, 4)

    # Plot subaxis data
    plot_subaxis(ax_array[0, 0], readout_a, "ChA readout")
    plot_subaxis(ax_array[0, 1], readout_b, "ChB readout")
    plot_iq(ax_array[0, 2], chA_nosub, chB_nosub, "I vs Q nosub")
    plot_iq(ax_array[0, 3], chA_sub, chB_sub, "I vs Q sub")
    plot_hist(ax_array[1, 0], chA_nosub, num_bins, "chA_nosub")
    plot_hist(ax_array[1, 1], chB_nosub, num_bins, "chB_nosub")
    plot_hist(ax_array[1, 2], mags_nosub, num_bins, "mags_nosub")
    plot_hist(ax_array[2, 0], chA_sub, num_bins, "chA_sub")
    plot_hist(ax_array[2, 1], chB_sub, num_bins, "chB_sub")
    plot_hist(ax_array[2, 2], mags_sub, num_bins, "mags_sub")
    fig.delaxes(ax_array[1, 3])
    fig.delaxes(ax_array[2, 3])

    if path:
        plt.savefig(path + "_pic", dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()

    # Plot pattern vs. voltage data
    (pattern_avgs_cA, pattern_avgs_cA_sub, pattern_avgs_cB, pattern_avgs_cB_sub, mags, mags_sub) = data.get_avgs()
    x = [i * time_step for i in range(num_patterns)]
    fig2, ax_array = plt.subplots(2, 3)
    font_size = 5

    plot_pattern_vs_volt(ax_array[0, 0], x, pattern_avgs_cA, "ChA nosub", font_size)
    plot_pattern_vs_volt(ax_array[0, 1], x, pattern_avgs_cB, "ChB nosub", font_size)
    plot_pattern_vs_volt(ax_array[1, 0], x, pattern_avgs_cA_sub, "ChA sub", font_size)
    plot_pattern_vs_volt(ax_array[1, 1], x, pattern_avgs_cB_sub, "ChB sub", font_size)
    plot_pattern_vs_volt(ax_array[0, 2], x, mags, "mags nosub", font_size)
    plot_pattern_vs_volt(ax_array[1, 2], x, mags_sub, "mags sub", font_size)

    if path:
        plt.savefig(path + "_pic2", dpi=300, pad_inches=0, bbox_inches='tight')
    plt.show()

# Define a function to plot various data representations
def plot_all(chA, chB, num_patterns, pattern_reps, seq_reps, avg_start, avg_length, large_data_plot=False):
    bins = 150
    cAp = get_p1_p2(chA, num_patterns, pattern_reps, seq_reps)
    cBp = get_p1_p2(chB, num_patterns, pattern_reps, seq_reps)

    if large_data_plot:
        ind = 0
        for i in range(0, len(cAp1), pattern_reps):
            plt.subplot(seq_reps, 4, (4 * ind) + 1)
            plt.pcolormesh(cAp1[i:i + pattern_reps])

            plt.subplot(seq_reps, 4, (4 * ind) + 3)
            plt.pcolormesh(cBp1[i:i + pattern_reps])

            if num_patterns == 2:
                plt.subplot(seq_reps, 4, (4 * ind) + 2)
                plt.pcolormesh(cAp2[i:i + pattern_reps])

                plt.subplot(seq_reps, 4, (4 * ind) + 4)
                plt.pcolormesh(cBp2[i:i + pattern_reps])
            ind += 1

        plt.figure()

    plt.subplot(3, 3, 1)
    ts = time.time()
    for i in range(num_patterns):
        plt.plot(average_all_iterations(cAp[i]))

    te = time.time()
    plt.title('channel A average p1 p2')

    plt.subplot(3, 3, 2)
    for i in range(num_patterns):
        plt.plot(average_all_iterations(cBp[i]))
    plt.title('channel B average p1 p2')

    plt.subplot(3, 3, 3)
    for i in range(num_patterns):
        cApn_av = get_avgs(cAp[i], avg_start, avg_length, False)
        cBpn_av = get_avgs(cBp[i], avg_start, avg_length, False)
        plt.scatter(cApn_av, cBpn_av, alpha=0.3)

    plt.title("I vs Q without subtraction")

    plt.subplot(3, 3, 4)
    s = time.time()
    for i in range(num_patterns):
        cApn_av = get_avgs(cAp[i], avg_start, avg_length)
        plt.hist(cApn_av, bins=bins, histtype='step')

    plt.title('chA with subtraction')
    plt.legend(["pattern " + str(i) for i in range(num_patterns)])

    plt.subplot(3, 3, 5)
    for i in range(num_patterns):
        cBpn_av = get_avgs(cBp[i], avg_start, avg_length)
        plt.hist(cBpn_av, bins=bins, histtype='step')

    plt.title('chB with subtraction')
    plt.legend(["pattern " + str(i) for i in range(num_patterns)])

    plt.subplot(3, 3, 6)
    for i in range(num_patterns):
        cApn_av = get_avgs(cAp[i], avg_start, avg_length, subtract=True)
        cBpn_av = get_avgs(cBp[i], avg_start, avg_length, subtract=True)
        mags = np.zeros(len(cApn_av))
        for j in range(len(cApn_av)):
            t_mag = np.sqrt(cApn_av[j] ** 2 + cBpn_av[j] ** 2)
            mags[j] = t_mag
        plt.hist(mags, bins=bins, histtype='step')
        plt.title("Magnitude with subtraction")

    plt.subplot(3, 3, 7)
    for i in range(num_patterns):
        cApn_av = get_avgs(cAp[i], avg_start, avg_length, subtract=False)
        plt.hist(cApn_av, bins=bins, histtype='step')
    plt.title("chA no subtraction")

    plt.subplot(3, 3, 8)
    for i in range(num_patterns):
        cBpn_av = get_avgs(cBp[i], avg_start, avg_length, subtract=False)
        plt.hist(cBpn_av, bins=bins, histtype='step')

    plt.title("chB no subtraction")

    plt.subplot(3, 3, 9)
    for i in range(num_patterns):
        cApn_av = get_avgs(cAp[i], avg_start, avg_length, subtract=False)
        cBpn_av = get_avgs(cBp[i], avg_start, avg_length, subtract=False)
        mags = np.zeros(len(cApn_av))
        for j in range(len(cApn_av)):
            t_mag = np.sqrt(cApn_av[j] ** 2 + cBpn_av[j] ** 2)
            mags[j] = t_mag
        plt.hist(mags, bins=bins, histtype='step')
        plt.title("Magnitude no subtraction")

    plt.show()

if __name__ == "__main__":
    num_patterns = 2
    pattern_reps = 10
    seq_reps = 3
    (chA, chB) = frombin(numAcquisitions=num_patterns * pattern_reps * seq_reps)
    plot_all(chA, chB, num_patterns, pattern_reps, seq_reps)
