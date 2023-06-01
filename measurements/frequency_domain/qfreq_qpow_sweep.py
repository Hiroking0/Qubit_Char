# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:16:01 2023

@author: lqc
"""
import sys
sys.path.append("../../")
import qcodes as qc
from instruments.Agilent_N5183A import N5183A
from qcodes.instrument_drivers.Keysight.N52xx import PNABase
from qcodes.dataset import (
    LinSweep,
    Measurement,
    dond,
    experiments,
    initialise_or_create_database_at,
    load_by_run_spec,
    load_or_create_experiment,
    plot_dataset,
    plot_by_id
)
import matplotlib.pyplot as plt
import numpy as np
from qcodes.parameters import Parameter


rf = N5183A('qubit_rf', "TCPIP0::172.20.1.7::5025::SOCKET")
rf.set('power', -22)
rf.set('frequency', 2.9396)
rf.set('enable', True)

VNA = PNABase(name = 'test',
              address = 'TCPIP0::K-E5080B-00202.local::hislip0::INSTR',
              min_freq = 9e6,
              max_freq = 9e9,
              min_power = -100,
              max_power = 20,
              nports = 2
              )
VNA.set('power', -65)
VNA.set('points',51)
VNA.set('timeout',None)
VNA.set('if_bandwidth',200)
VNA.set('cw', 7.20263e9)
VNA.set('trace','S21')
VNA.set('sweep_type', 'CW')
#VNA.get_idn()
station = qc.Station()
station.add_component(rf)
station.add_component(VNA)
pow_start = 12
pow_stop = 15
pow_step = 1

freq_start = 3.345
freq_stop = 3.355
freq_step = .0004



initialise_or_create_database_at("./databases/wq_pq_sweep.db")

tutorial_exp = load_or_create_experiment(
    experiment_name="tutorial_exp",
    sample_name="synthetic data"
)

context_meas = Measurement(exp=tutorial_exp, station=station, name='context_example')
# Register the independent parameter...

param1 = rf.power
param2 = rf.frequency
context_meas.register_parameter(param1)
context_meas.register_parameter(param2)
z = Parameter(name='z', label='Magnitude', unit='dB',
              set_cmd=None, get_cmd=None)

context_meas.register_parameter(z, setpoints = (param1, param2))

with context_meas.run() as datasaver:
    for pow_set in np.arange(pow_start, pow_stop, pow_step):
        param1.set(pow_set)
        print(pow_set)
        for freq_set in np.arange(freq_start, freq_stop, freq_step):
            print(freq_set)
            param2.set(freq_set)
            mag = np.average(VNA.magnitude())
            datasaver.add_result((param1, pow_set), (param2, freq_set), (z, mag))
            dataid = datasaver.run_id
            dataset = datasaver.dataset
    
rf.set('enable', False)
plot_dataset(dataset)

plt.show()

