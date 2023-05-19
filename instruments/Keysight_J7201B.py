# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:54:48 2023

@author: lqc
"""
from qcodes.instrument import VisaInstrument
from typing import Any
from qcodes.validators import Numbers

class J7201B(VisaInstrument):
    def __init__(self,
                 name: str,
                 address: str,
                 **kwargs: Any):
                 # Set frequency ranges
        super().__init__(name, address, terminator='\n', **kwargs)
        
        self.add_parameter(
            name = "attenuation",
            label="Attenuation",
            unit = 'dB',
            get_cmd=":ATT?",
            set_cmd=":Att {} dB",
            vals = Numbers(min_value = 0, max_value = 121)
        )