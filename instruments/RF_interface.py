# Import the base class for the instrument
from .LAN_instrument_base import Instrument

class RF_source(Instrument):

    def set_freq(self, freq: float):
        """
        Set the frequency of the RF source.

        Args:
            freq (float): Frequency in GHz to be set.
        """
        command = ':FREQuency:CW ' + str(freq) + 'GHz'
        print(command)  # Print the command for reference
        return self.inst.write(command)

    def set_power(self, power: float):
        """
        Set the output power of the RF source.

        Args:
            power (float): Power level in dBm to be set.
        """
        command = ':POWer:AMPLitude ' + str(power) + 'DBM'
        return self.inst.write(command)

    def enable_out(self):
        """
        Enable the RF output.
        """
        self.inst.write(':OUTPut:STATe ON')

    def disable_out(self):
        """
        Disable the RF output.
        """
        self.inst.write(':OUTPut:STATe OFF')
