# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:47:39 2023

@author: lqc
"""
# Import the base class for the instrument
from .LAN_instrument_base import Instrument

class Atten(Instrument):

    def set_attenuation(self, atten):
        """
        Set the attenuation level of the attenuator.

        Args:
            atten (float): Attenuation level in dB to be set.
        """
        command = ':ATT ' + str(atten) + 'dB'
        return self.inst.write(command)

    