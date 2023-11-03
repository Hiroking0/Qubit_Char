# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:16:01 2023

@author: lqc
"""
import qcodes as qc
import sys
sys.path.append("../../")
from instruments.Agilent_N5183A import N5183A
from qcodes.instrument_drivers.Keysight.N52xx import PNABase
from qcodes.dataset import (
    Measurement,
    initialise_or_create_database_at,
    load_or_create_experiment,
    plot_dataset
)
import matplotlib.pyplot as plt
import numpy as np
import tkinter.filedialog as filedialog

from datetime import datetime

now = datetime.now()

formatted_date = now.strftime("%Y_%m_%d_%H_%M_%S")

name = 'resonator_spec_'+formatted_date
rf = N5183A('qubit_rf', "TCPIP0::172.20.1.7::5025::SOCKET")
VNA = PNABase(name = 'test',
              address = 'TCPIP0::K-E5080B-00202.local::hislip0::INSTR',
              min_freq = 9e6,
              max_freq = 9e9,
              min_power = -100,
              max_power = 20,
              nports = 2
              )

VNA.set('start',7.0879e9)
VNA.set('stop',7.0883e9)
VNA.set('points',1001)
VNA.set('timeout',None)
VNA.set('if_bandwidth',50)
VNA.set('averages_enabled', False)
VNA.set('averages', 1) #Only works for sweep mode averaging
VNA.set('trace','S21')
VNA.set('sweep_type', 'LOG')
print(VNA.get_idn())

station = qc.Station()
station.add_component(rf)
station.add_component(VNA)

sweep_start = -70
sweep_stop = -40
sweep_step = 1
path = filedialog.askdirectory() + "/" + name + "_"
initialise_or_create_database_at(path)

tutorial_exp = load_or_create_experiment(
    experiment_name="Resonator Spectroscopy",
    sample_name="qubit 4"
)

context_meas = Measurement(exp=tutorial_exp, station=station, name='context_example')
# Register the independent parameter...

param = VNA.power
context_meas.register_parameter(param)

#context_meas.register_parameter(rf.frequency)
context_meas.register_parameter(VNA.magnitude, setpoints = (param,))

with context_meas.run() as datasaver:
    for set_param in np.arange(sweep_start, sweep_stop, sweep_step):
        print(set_param)
        VNA.set('power', set_param)
        #rf.frequency.set(set_freq)
        mag = VNA.magnitude()
        datasaver.add_result((VNA.magnitude, mag), (param, set_param))
dataid = datasaver.run_id
dataset = datasaver.dataset
dataset.export("netcdf", path=path+'actualdata') 
plot_dataset(dataset)

plt.show()

