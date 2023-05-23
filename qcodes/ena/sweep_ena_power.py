# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:16:01 2023

@author: lqc
"""
import qcodes as qc
from RF_qcodes import AgilentN5183A
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



rf = AgilentN5183A('qubit_rf', "TCPIP0::172.20.1.7::5025::SOCKET")
VNA = PNABase(name = 'test',
              address = 'TCPIP0::K-E5080B-00202.local::hislip0::INSTR',
              min_freq = 9e6,
              max_freq = 9e9,
              min_power = -100,
              max_power = 20,
              nports = 2
              )

VNA.set('start',7.08335e9)
VNA.set('stop',7.08535e9)
VNA.set('points',1001)
VNA.set('timeout',None)
VNA.set('if_bandwidth',200)
VNA.set('trace','S21')
VNA.set('sweep_type', 'LOG')
VNA.get_idn()
station = qc.Station()
station.add_component(rf)
station.add_component(VNA)
sweep_start = -70
sweep_stop = -20
sweep_step = 5

initialise_or_create_database_at("./ena_power.db")

tutorial_exp = load_or_create_experiment(
    experiment_name="tutorial_exp",
    sample_name="synthetic data"
)

context_meas = Measurement(exp=tutorial_exp, station=station, name='context_example')
# Register the independent parameter...

param = VNA.power
context_meas.register_parameter(param)

#context_meas.register_parameter(rf.frequency)
context_meas.register_parameter(VNA.magnitude, setpoints = (param,))

with context_meas.run() as datasaver:
    for set_param in np.arange(sweep_start, sweep_stop, sweep_step):
        VNA.set('power', set_param)
        #rf.frequency.set(set_freq)
        mag = VNA.magnitude()
        datasaver.add_result((VNA.magnitude, mag), (param, set_param))
        dataid = datasaver.run_id
        dataset = datasaver.dataset
    
plot_dataset(dataset)

#plot_by_id(dataid)
plt.show()

# ...then register the dependent parameter
#context_meas.register_parameter(dmm.v1, setpoints=(dac.ch1,))

