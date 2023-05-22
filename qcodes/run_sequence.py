# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:01:42 2023

@author: lqc
"""

import qcodes as qc
import matplotlib.pyplot as plt
import qcodes.instrument_drivers.AlazarTech as ats
from ac_controller import qubit_ac_controller, set_alazar_settings
import yaml
from lib.wave_construction import get_awg
import sys
sys.path.append("../")
from instruments import Agilent_N5183A
from instruments import Keysight_J7201B

from qcodes.dataset import (
    LinSweep,
    Measurement,
    dond,
    experiments,
    initialise_or_create_database_at,
    load_by_run_spec,
    load_or_create_experiment,
    plot_dataset,
)


if __name__ == "__main__":
    
    f = open('../measurements/general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()

    awg = get_awg()
    alazar_name = "t_name"
    at = ats.AlazarTechATS9870(alazar_name, dll_path = 'C:\\WINDOWS\\System32\\ATSApi.dll')
    ac = set_alazar_settings(params, at, awg)
    ac = qubit_ac_controller('t_ac',
                             alazar_name,
                             params['avg_start'],
                             params['avg_length'],
                             params['pattern_repeat'],
                             params['seq_repeat'],
                             awg.get_seq_length(),
                             awg)
    q_rf = Agilent_N5183A.N5183A("q_rf", "TCPIP0::172.20.1.7::5025::SOCKET")
    r_rf = Agilent_N5183A.N5183A("r_rf", "TCPIP0::172.20.1.8::5025::SOCKET")
    q_att = Keysight_J7201B.J7201B("q_att", "TCPIP0::172.20.1.6::5025::SOCKET")
    r_att = Keysight_J7201B.J7201B("r_att", "TCPIP0::172.20.1.9::5025::SOCKET")
    
    station = qc.Station()
    station.add_component(q_rf)
    station.add_component(r_rf)
    station.add_component(q_att)
    station.add_component(r_att)
    
    initialise_or_create_database_at("./database.db")
    
    
    tutorial_exp = load_or_create_experiment(
    experiment_name="tutorial_exp",
    sample_name="synthetic data"
    )
    
    context_meas = Measurement(exp=tutorial_exp, station=station, name='context_example')
    context_meas.register_parameter(ac.acquisition)
    
    with context_meas.run() as datasaver:
        for set_v in np.linspace(0, 25, 10):
            dac.ch1.set(set_v)
            get_v = dmm.v1.get()
            datasaver.add_result((dac.ch1, set_v),
                                 (dmm.v1, get_v))

        # Convenient to have for plotting and data access
        dataset = datasaver.dataset
    









    
    
    (chA_sub, chB_sub, chA_nosub, chB_nosub) = at.acquire(acquisition_controller = ac)
    plt.hist(chB_nosub[0], bins = 200, histtype = 'step')
    plt.hist(chB_nosub[1], bins = 200, histtype = 'step')
    plt.show()