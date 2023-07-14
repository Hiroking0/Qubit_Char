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
from lib import wave_construction as wc
rf = N5183A('qubit_rf', "TCPIP0::172.20.1.7::5025::SOCKET")
param1 = rf.power
param2 = rf.frequency

awg = wc.get_awg()


VNA = PNABase(name = 'test',
              address = 'TCPIP0::K-E5080B-00202.local::hislip0::INSTR',
              min_freq = 9e6,
              max_freq = 9e9,
              min_power = -100,
              max_power = 20,
              nports = 2
              )
VNA.set('power', -55)
VNA.set('points',51)
VNA.set('timeout',None)
VNA.set('if_bandwidth',200)
VNA.set('cw', 3.25e9)
VNA.set('trace','S21')
VNA.set('sweep_type', 'CW')
VNA.set('averages_enabled', False)
#VNA.get_idn()
station = qc.Station()
station.add_component(rf)
station.add_component(VNA)
c1off_start = -2
c1off_stop = 2
c1off_step = .05

c2off_start = -2
c2off_stop = 2
c2off_step = .05


initialise_or_create_database_at("./databases/offset_sweep.db")

tutorial_exp = load_or_create_experiment(
    experiment_name="tutorial_exp",
    sample_name="synthetic data"
)

context_meas = Measurement(exp=tutorial_exp, station=station, name='context_example')
# Register the independent parameter...


context_meas.register_parameter(param1)
context_meas.register_parameter(param2)
z = Parameter(name='z', label='Magnitude', unit='dB',
              set_cmd=None, get_cmd=None)

context_meas.register_parameter(z, setpoints = (param1, param2))

with context_meas.run() as datasaver:
    for o1 in np.arange(c1off_start, c1off_stop, c1off_step):
        #param1.set(pow_set)
        awg.set_offset(o1, 1)
        print("o1", o1)
        for o2 in np.arange(c2off_start, c2off_stop, c2off_step):
            awg.set_offset(o2, 2)
            print("o2", o2)
            #param2.set(freq_set)
            mag = np.average(VNA.magnitude())
            datasaver.add_result((param1, o1), (param2, o2), (z, mag))
            dataid = datasaver.run_id
            dataset = datasaver.dataset
    
#rf.set('enable', False)
plot_dataset(dataset)

plt.show()

