# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:01:42 2023

@author: lqc
"""

import matplotlib.pyplot as plt
import qcodes.instrument_drivers.AlazarTech as ats
from ac_controller import qubit_ac_controller
import yaml
from lib.wave_construction import get_awg


def set_alazar_settings(params, alazar, awg):
    decimation = params['decimation']
    seq_repeat = params['seq_repeat']
    pattern_repeat = params['pattern_repeat']
    acq_multiples = params['acq_multiples']
    avg_start = params['avg_start']
    avg_len = params['avg_length']
    at.set('mode', 'NPT')
    at.set('clock_source', 'EXTERNAL_CLOCK_10MHz_REF')
    at.set('decimation', decimation)
    at.set('coupling1', 'DC')
    at.set('coupling2', 'DC')
    sensitivity = .1
    at.set('channel_range1', sensitivity)
    at.set('channel_range2', sensitivity)
    
    at.set('trigger_source1', 'EXTERNAL')
    at.set('trigger_level1', 150)
    rec_mult = 256*acq_multiples
    
    at.set('samples_per_record', rec_mult)
    
    num_patterns = awg.get_seq_length()
    at.set('buffers_per_acquisition', seq_repeat * pattern_repeat * num_patterns)
    at.set('allocated_buffers', 6)
    at.set('buffer_timeout', 5000)
    at.sync_settings_to_card()


if __name__ == "__main__":
    
    f = open('../measurements/general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()

    awg = get_awg()
    alazar_name = "t_name"
    at = ats.AlazarTechATS9870(alazar_name, dll_path = 'C:\\WINDOWS\\System32\\ATSApi.dll')
    ac = set_alazar_settings(params, at, awg)
    #self, name, alazar_name, avg_start, avg_len, pattern_rep, seq_rep, awg, **kwargs)
    ac = qubit_ac_controller('t_ac',
                             alazar_name,
                             params['avg_start'],
                             params['avg_length'],
                             params['pattern_repeat'],
                             params['seq_repeat'],
                             awg.get_seq_length(),
                             awg)
    
    (chA_sub, chB_sub, chA_nosub, chB_nosub) = at.acquire(acquisition_controller = ac)
    plt.hist(chA_nosub[0], bins = 200, histtype = 'step')
    plt.hist(chA_nosub[1], bins = 200, histtype = 'step')
    plt.show()