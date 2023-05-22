# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:01:42 2023

@author: lqc
"""

import matplotlib.pyplot as plt
import qcodes.instrument_drivers.AlazarTech as ats
from ac_controller import qubit_ac_controller, set_alazar_settings
import yaml
from lib.wave_construction import get_awg


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
    
    (chA_sub, chB_sub, chA_nosub, chB_nosub) = at.acquire(acquisition_controller = ac)
    plt.hist(chB_nosub[0], bins = 200, histtype = 'step')
    plt.hist(chB_nosub[1], bins = 200, histtype = 'step')
    plt.show()