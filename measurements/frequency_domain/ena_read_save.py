# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:16:39 2023

@author: lqc
"""
from qcodes.instrument_drivers.Keysight.N52xx import PNABase
#import numpy as np
import qcodes as qc
import pyvisa as visa
from matplotlib import pyplot as plt
import time
import csv
import tkinter as tk
from datetime import datetime
from tkinter.filedialog import askdirectory
import tkinter.filedialog as filedialog


from qcodes.dataset import (
    Measurement,
    initialise_or_create_database_at,
    load_or_create_experiment,
    plot_dataset,
    plot_by_id
)
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.signal import peak_prominences


def objective_lorentz(x, a, b, c, f, g):
    return a + b/(c*(x+f)**2 + g)

def fit_lor(x, y, init_a, init_b, init_c, init_f, init_g):
    #x_data = np.linspace(0, max_length, num_points)
    #y_data = y
    initial  = [init_a, init_b, init_c, init_f, init_g]
    popt, _ = curve_fit(objective_lorentz, x, y, p0 = initial)
    a, b, c, f, g = popt
    
    new_data = objective_lorentz(x, a, b, c, f, g)
    return new_data, a, b, c, f, g







'''now = datetime.now()
Date = now.strftime("%Y%m%d_%H%M%S")'''





VNA = PNABase(name = 'test',
              address = 'TCPIP0::K-E5080B-00202.local::hislip0::INSTR',
              min_freq = 9e6,
              max_freq = 9e9,
              min_power = -100,
              max_power = 20,
              nports = 2
              )


#print(VNA.get_options())
probe_start  = 7.221e9
probe_stop   = 7.222e9

probe_pwr = -3 # in dBm
num_points=2001
if_bandwidth=500
#timeout=100000
avgs = 1

name = 'read_save.db'
path = filedialog.askdirectory() + "/" + name + "_"
print(path)
initialise_or_create_database_at(path)

VNA.set('power',probe_pwr)
VNA.set('start',probe_start)
VNA.set('stop',probe_stop)
VNA.set('points',num_points)
#VNA.set('timeout',timeout)
VNA.set('if_bandwidth',if_bandwidth)
VNA.set('trace','S21')
VNA.set('sweep_type', 'LOG')
VNA.set('averages_enabled', True)
VNA.set('averages', avgs) #Only works for sweep mode averaging
#print(VNA.get_idn())
station = qc.Station()
station.add_component(VNA)


tutorial_exp = load_or_create_experiment(experiment_name="save_ena")

context_meas = Measurement(exp=tutorial_exp, station=station, name='res_spec')

#param = VNA.phase, VNA.magnitude

context_meas.register_parameter(VNA.phase)
context_meas.register_parameter(VNA.magnitude)
context_meas.register_parameter(VNA.real)
context_meas.register_parameter(VNA.imaginary)


with context_meas.run() as datasaver:
    phase = VNA.phase()
    mag = VNA.magnitude()
    real = VNA.real()
    imag = VNA.imaginary()
    datasaver.add_result((VNA.phase, phase), (VNA.magnitude, mag), (VNA.real, real), (VNA.imaginary, imag))
    dataset = datasaver.dataset
#plot_dataset(dataset)


### Begin data saving procedure
import tkinter as tk
import os
import glob
from tkinter import filedialog
from datetime import datetime

file_path = filedialog.askdirectory() # Variable to prompt the user to select a file directory
dataset.export('csv', path = file_path) # Set the file type and path as file_path... this will open the file explorer and allow the user to select their desired save location.


### File name formating and then renaming
fileName = 'start_' + str(probe_start/1e9) + '_stop_' + str(probe_stop/1e9) + '_pow_' + str(probe_pwr)

now = datetime.now()
Date = now.strftime("%Y%m%d_%H%M%S")

file_name_format = Date + ' ' + fileName + '.txt'

### Renaming procedure
path = os.path.join(file_path)
path_a = path + "/*"
list_of_files = glob.glob(path_a) # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)

new_file = os.path.join(path, file_name_format)
print(latest_file) # prints a.txt which was latest file i created
os.rename(latest_file, new_file)

### Print qCoDeS export information
print(dataset.export_info)

print(new_file)
#plt.show()

with open(new_file) as f:
    output = [float(x) for x in f.read().split()]
#print(output)


freq_raw = output[0::5]
real_raw = output[4::5]
imag_raw = output[1::5]

mag_raw = []
phase_raw = []
for i in range(0, len(real_raw)):
    mag_temporary = np.sqrt(real_raw[i]*real_raw[i] + imag_raw[i]*imag_raw[i])
    mag_tempory_log = 20*np.log10(mag_temporary)
    mag_raw.append(mag_tempory_log)

    phase_temporary = np.arctan(imag_raw[i]/real_raw[i])
    phase_raw.append(phase_temporary)

#print(mag_raw)
#print(phase_raw)


label = ["Magnitude", "Real", "Imaginary", "Phase"]
ylabel = ["Magnitude (dBm)", "Real (dBm)", "Imaginary (dBm)", "Phase (Degrees)"]
data = [mag_raw, real_raw, imag_raw, phase_raw]

plt.figure(figsize=(10,10))

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(freq_raw, data[i], label=label[i], color = 'black')
    plt.xlabel('frequency (GHz)')
    plt.ylabel(ylabel[i])
plt.show()

plt.plot(real_raw, imag_raw, color = 'black')
plt.xlabel('Real (GHz)')
plt.ylabel('Imaginary (GHz)')
plt.show()

































'''x = np.linspace(probe_start, probe_stop, num = num_points)/1e9
#peaks, _= find_peaks(values*-1, height = 75)
peaks, _= find_peaks(values, prominence = 10)

#proms = peak_prominences(values*-1, peaks)
#print("prominences", proms)
print(peaks)

f = -x[peaks[0]]
print(f)
#plot_dataset(dataset)
a = -57
b = -30
c = 100_000_000
#f = -7.2017
g = 1


guess = objective_lorentz(x, a, b, c, f, g)
fit_data, new_a, new_b, new_c, new_f, new_g = fit_lor(x, values, a, b, c, f, g)
print(new_a, new_b, new_c, new_f, new_g)
#plt.plot(x, guess)
plt.plot(x, values, 'ko', markersize=10)
plt.plot(x, fit_data, 'r', linewidth=3.5)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude")
plt.show()
'''




















'''now = datetime.now()
Date = now.strftime("%Y%m%d_%H%M%S")
root = tk.Tk()
pathName = askdirectory(title='Select Folder')
root.withdraw()

file = "".join((pathName, '\\', Date, ' ', fileName, '.txt'))
with open(file, 'w', newline='') as output:

    wr = csv.writer(output, delimiter='\t', quoting=csv.QUOTE_NONE)
    header=['Frequency (GHz)', 'Real', 'Imag']
    wr.writerow(header)
    for i in range(0,len(mag_raw)):

        wr.writerow([freqs[i], '{:3.5e}'.format(real_raw[i]), '{:3,5e}'.format(imag_raw[i])])
    output.write('\n')

print('Done')'''

