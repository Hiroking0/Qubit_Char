# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:16:39 2023

@author: lqc
"""
from qcodes.instrument_drivers.Keysight.N52xx import PNABase
import numpy as np
from matplotlib import pyplot as plt

VNA = PNABase(name = 'test',
              address = 'TCPIP0::K-E5080B-00202.local::hislip0::INSTR',
              min_freq = 9e6,
              max_freq = 9e9,
              min_power = -100,
              max_power = 20,
              nports = 2
              )
print(VNA.get_options())
probe_start  = 7.197e9
probe_stop   = 7.205e9

probe_pwr = -60 # in dBm
num_points=1001
if_bandwidth=100
timeout=100000

VNA.set('power',probe_pwr)
VNA.set('start',probe_start)
VNA.set('stop',probe_stop)
VNA.set('points',num_points)
VNA.set('timeout',timeout)
VNA.set('if_bandwidth',if_bandwidth)
VNA.set('trace','S21')
VNA.set('sweep_type', 'LOG')
VNA.get_idn()


#VNA.traces.tr1.run_sweep()

#values=VNA.real()+1j*VNA.imaginary()
values = VNA.magnitude()
coordinates=np.linspace(probe_start,probe_stop,num_points)
plt.plot(coordinates/1E9, values)
plt.xlabel('Frequency (GHz)')
plt.ylabel('|$S_{21}$|')
plt.show()


