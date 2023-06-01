# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:16:39 2023

@author: lqc
"""
from qcodes.instrument_drivers.Keysight.N52xx import PNABase
#import numpy as np
import qcodes as qc
from matplotlib import pyplot as plt
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


VNA = PNABase(name = 'test',
              address = 'TCPIP0::K-E5080B-00202.local::hislip0::INSTR',
              min_freq = 9e6,
              max_freq = 9e9,
              min_power = -100,
              max_power = 20,
              nports = 2
              )
#print(VNA.get_options())
probe_start  = 7.2014e9
probe_stop   = 7.2036e9

probe_pwr = -55 # in dBm
num_points=1001
if_bandwidth=500
#timeout=100000
avgs = 5

initialise_or_create_database_at("./databases/read_save.db")

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


tutorial_exp = load_or_create_experiment(
    experiment_name="save_ena"
)

context_meas = Measurement(exp=tutorial_exp, station=station, name='res_spec')
param = VNA.magnitude
context_meas.register_parameter(param)


with context_meas.run() as datasaver:
    values = VNA.magnitude()
    datasaver.add_result((VNA.magnitude, values))
    dataset = datasaver.dataset
plot_dataset(dataset)
plt.show()


x = np.linspace(probe_start, probe_stop, num = num_points)/1e9
#peaks, _= find_peaks(values*-1, height = 75)
peaks, _= find_peaks(values*-1, prominence = 10)

proms = peak_prominences(values*-1, peaks)
print("prominences", proms)
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
#x = range(len(values))
fit_data, new_a, new_b, new_c, new_f, new_g = fit_lor(x, values, a, b, c, f, g)
print(new_a, new_b, new_c, new_f, new_g)
#plt.plot(x, guess)
plt.plot(x, values, 'ko', markersize=10)
plt.plot(x, fit_data, 'r', linewidth=3.5)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude")
plt.show()






















