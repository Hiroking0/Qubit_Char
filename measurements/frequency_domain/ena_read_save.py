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

VNA = PNABase(name = 'test',
              address = 'TCPIP0::K-E5080B-00202.local::hislip0::INSTR',
              min_freq = 9e6,
              max_freq = 9e9,
              min_power = -100,
              max_power = 20,
              nports = 2
              )
#print(VNA.get_options())
probe_start  = 7.197e9
probe_stop   = 7.205e9

probe_pwr = -60 # in dBm
num_points=1001
if_bandwidth=500
timeout=100000

initialise_or_create_database_at("./databases/read_save.db")

VNA.set('power',probe_pwr)
VNA.set('start',probe_start)
VNA.set('stop',probe_stop)
VNA.set('points',num_points)
VNA.set('timeout',timeout)
VNA.set('if_bandwidth',if_bandwidth)
VNA.set('trace','S21')
VNA.set('sweep_type', 'LOG')
VNA.get_idn()
station = qc.Station()
station.add_component(VNA)


tutorial_exp = load_or_create_experiment(
    experiment_name="tutorial_exp",
    sample_name="synthetic data"
)

context_meas = Measurement(exp=tutorial_exp, station=station, name='context_example')
param = VNA.magnitude
context_meas.register_parameter(param)

#context_meas.register_parameter(rf.frequency)
#context_meas.register_parameter(VNA.magnitude, setpoints = (param,))


#VNA.traces.tr1.run_sweep()

#values=VNA.real()+1j*VNA.imaginary()
with context_meas.run() as datasaver:
    values = VNA.magnitude()
    datasaver.add_result((VNA.magnitude, values))
    #dataid = datasaver.run_id
    dataset = datasaver.dataset

plot_dataset(dataset)
plt.show()


