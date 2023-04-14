# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:47:39 2023

@author: lqc
"""
from .LAN_instrument_base import Instrument


class Atten(Instrument):

    def set_attenuation(self, atten):
        command = ':ATT ' + str(atten) + 'dB'
        return self.inst.write(command)
    
    