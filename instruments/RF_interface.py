
#import pyvisa as visa
from .LAN_instrument_base import Instrument

class RF_source(Instrument):

    #freq in MHz
    def set_freq(self, freq: float):
        command = ':FREQuency:CW ' + str(freq) + 'Hz'
        print(command)
        return self.inst.write(command)

    def set_power(self, power: float):
        command = ':POWer:AMPLitude ' + str(power) + 'DBM'
        return self.inst.write(command)
    
    def enable_out(self):
        self.inst.write(':OUTPut:STATe ON')
        
    def disable_out(self):
        self.inst.write(':OUTPut:STATe OFF')