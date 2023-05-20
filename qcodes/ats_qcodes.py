# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:00:47 2023

@author: lqc
"""
import sys
sys.path.append("../")

import qcodes as qc
import qcodes.instrument_drivers.AlazarTech as ats
from qcodes.instrument_drivers.AlazarTech import AcquisitionController
import numpy as np
from typing import Any, Dict, Optional, Tuple, TypeVar
from lib import wave_construction as be
OutputType = TypeVar('OutputType')


class qubit_ac_controller(AcquisitionController[float]):
    
    """
    This class represents all choices that the end-user has to make regarding
    the data-acquisition. this class should be subclassed to program these
    choices.

    The basic structure of an acquisition is:

        - Call to :meth:`AlazarTech_ATS.acquire` internal configuration
        - Call to :meth:`AcquisitionInterface.pre_start_capture`
        - Call to the start capture of the Alazar board
        - Call to :meth:`AcquisitionInterface.pre_acquire`
        - Loop over all buffers that need to be acquired
          dump each buffer to acquisitioncontroller.handle_buffer
          (only if buffers need to be recycled to finish the acquisiton)
        - Dump remaining buffers to :meth:`AcquisitionInterface.handle_buffer`
          alazar internals
        - Return return value from :meth:`AcquisitionController.post_acquire`
    """
    def __init__(self, name, alazar_name, num_patterns, avg_start, avg_len, **kwargs):
        self.num_patterns = num_patterns
        self.avg_start = avg_start
        self.avg_len = avg_len
        self.samples_per_record = 0
        self.records_per_buffer = 0
        self.buffers_per_acquisition = 0
        self.number_of_channels = 2
        self.chA_nosub: Optional[np.ndarray] = None
        self.chB_nosub: Optional[np.ndarray] = None
        self.chA_sub: Optional[np.ndarray] = None
        self.chB_sub: Optional[np.ndarray] = None
        self.awg = None
        self.acquisitionkwargs: Dict[str, Any] = {}
        super().__init__(name, alazar_name, **kwargs)



    def update_acquisitionkwargs(self, **kwargs: Any) -> None:
        """
        This method must be used to update the kwargs used for the acquisition
        with the alazar_driver.acquire
        :param kwargs:
        :return:
        """
        self.acquisitionkwargs.update(**kwargs)


    def pre_start_capture(self) -> None:
        """
        Use this method to prepare yourself for the data acquisition
        The Alazar instrument will call this method right before
        'AlazarStartCapture' is called
        """

        alazar = self._get_alazar()
        self.samples_per_record = alazar.samples_per_record.get()
        self.records_per_buffer = alazar.records_per_buffer.get()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        #sample_speed = alazar.get_sample_rate()
        self.chA_nosub = np.zeros((self.num_patterns, seq_repeat * pattern_repeat))
        self.chB_nosub = np.zeros((self.num_patterns, seq_repeat * pattern_repeat))
        self.chA_sub = np.zeros((self.num_patterns, seq_repeat * pattern_repeat))
        self.chB_sub = np.zeros((self.num_patterns, seq_repeat * pattern_repeat))

    def pre_acquire(self) -> None:
        """
        This method is called immediately after 'AlazarStartCapture' is called
        """
        self.awg = be.get_awg()
        self.awg.run()


    def handle_buffer(
        self, buffer: np.ndarray, buffer_number: int | None = None
        ) -> None:
        """
        This method should store or process the information that is contained
        in the buffers obtained during the acquisition.

        Args:
            buffer: np.array with the data from the Alazar card
            buffer_number: counter for which buffer we are handling

        """
        pattern_number = int(buffer_number/self.pattern_repeat) % self.num_patterns
        seq_number = int(buffer_number/(self.num_patterns*pattern_repeat))
        
        index_number = seq_number*self.pattern_repeat + buffer_number % self.pattern_repeat

        half = int(len(buffer)/2)
        chA = buffer[:half]
        chB = buffer[half:]
        t_Aavg = np.average(chA[self.avg_start: self.avg_start + self.avg_len])
        t_Bavg = np.average(chB[self.avg_start: self.avg_start + self.avg_len])
            
        self.chA_sub[pattern_number][index_number] = t_Aavg - np.average(chA[200:800])
        self.chB_sub[pattern_number][index_number] = t_Bavg - np.average(chB[200:800])
        self.chA_nosub[pattern_number][index_number] = t_Aavg
        self.chB_nosub[pattern_number][index_number] = t_Bavg

    def post_acquire(self) -> OutputType:
        """
        This method should return any information you want to save from this
        acquisition. The acquisition method from the Alazar driver will use
        this data as its own return value

        Returns:
            this function should return all relevant data that you want
            to get form the acquisition
        """
        self.awg.stop()
        self.awg.close()

        assert self.chA_sub is not None
        alazar = self._get_alazar()
        #convert to volts
        for i in range(len(self.chA_sub)):
            for j in range(len(self.chA_sub[0])):
                self.chA_sub[i,j] = alazar.signal_to_volt(1, self.chA_sub[i,j])
                self.chB_sub[i,j] = alazar.signal_to_volt(2, self.chB_sub[i,j])
                self.chA_nosub[i,j] = alazar.signal_to_volt(1, self.chA_nosub[i,j])
                self.chB_nosub[i,j] = alazar.signal_to_volt(2, self.chB_nosub[i,j])

        return (self.chA_sub, self.chB_sub, self.chA_nosub, self.chB_nosub)

at = ats.AlazarTechATS9870("t_name", dll_path = 'C:\\WINDOWS\\System32\\ATSApi.dll')

#print(at.get_idn())
at.set('mode', 'NPT')
at.set('clock_source', 'EXTERNAL_CLOCK_10MHz_REF')
at.set('decimation', 1)
at.set('coupling1', 'DC')
at.set('coupling2', 'DC')
sensitivity = .1
at.set('channel_range1', sensitivity)
at.set('channel_range2', sensitivity)

at.set('trigger_source1', 'EXTERNAL')
at.set('trigger_level1', 150)
rec_mult = 256*100
seq_repeat = 1000
pattern_repeat = 1
at.set('samples_per_record', rec_mult)
at.set('buffers_per_acquisition', seq_repeat * pattern_repeat)
at.set('allocated_buffers', 6)
at.set('buffer_timeout', 5000)
at.sync_settings_to_card()
ac = qubit_ac_controller('t_ac', "t_name")

(chA_sub, chB_sub, chA_nosub, chB_nosub) = at.acquire(acquisition_controller = ac)
