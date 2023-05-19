# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:00:47 2023

@author: lqc
"""

import qcodes as qc
import qcodes.instrument_drivers.AlazarTech as ats
from qcodes.instrument_drivers.AlazarTech import AcquisitionController

class qubit_ac_controller(AcquisitionController[float]):
    
    




at = ats.AlazarTechATS9870("t_name", dll_path = 'C:\\WINDOWS\\System32\\ATSApi.dll')

#print(at.get_idn())
at.set('mode', 'NPT')
at.set('clock_source', 'EXTERNAL_CLOCK_10MHz_REF')
at.set('decimation', 1)
at.set('coupling1', 'DC')
at.set('coupling2', 'DC')
at.set('channel_range1', .1)
at.set('channel_range2', .1)

at.set('trigger_source1', 'EXTERNAL')
at.set('trigger_level1', 150)
rec_mult = 256*100
at.set('samples_per_record', rec_mult)
at.set('buffers_per_acquisition', 1000)
at.set('allocated_buffers', 6)
at.set('buffer_timeout', 5000)


at.sync_settings_to_card()
ac = AcquisitionController('t_ac', "t_name")


at.acquire(acquisition_controller = ac)