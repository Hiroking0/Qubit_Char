# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:35:40 2023

@author: lqc
"""
from typing import Any
from qcodes.instrument import VisaInstrument
from qcodes.validators import Numbers

class AgilentN5183A(VisaInstrument):
    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address, terminator="\n", **kwargs)
        self.add_parameter(
            name="frequency",
            label="Frequency",
            unit="GHz",
            get_cmd=":FREQuency:CW?",
            set_cmd=":FREQuency:CW {} GHz",
            get_parser=float,
            vals=Numbers(min_value=.0001, max_value=20),
        )
        
        self.add_parameter(
            name="power",
            label="Power",
            unit="DBM",
            get_cmd=":POWer:AMPLitude?",
            set_cmd=":POWer:AMPLitude {}",
            get_parser=float,
            vals=Numbers(min_value=-32, max_value=30),
        )
        
        