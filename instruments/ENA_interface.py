"""
Hiroki Fujisato

Class for the Keysight E5080A ENA Network Analyzer

"""
# Importing the necessary base class for the instrument
from .LAN_instrument_base import Instrument

class ENA(Instrument):

    def set_power(self, power: float):
        """Set the power for the ENA

        Args:
            power (float): Power level in dBm
        """
        self.inst.write('SOUR1:POW ' + str(power))

    def set_freq_range(self, startfreq: str, stopfreq: str):
        """Set the sweep frequency range in GHz

        Args:
            startfreq (str): Start frequency of the sweep
            stopfreq (str): End frequency of the sweep
        """
        self.inst.write('SENS1:FREQ:STAR ' + startfreq + 'E9')
        self.inst.write('SENS1:FREQ:STOP ' + stopfreq + 'E9')

    # Set the averaging mode and number of averages
    def set_averaging(self, mode, num: int):
        """Set the averaging mode and count

        Args:
            mode (str): Averaging mode ('POINt' or 'SWEEP')
            num (int): Number of averages
        """
        self.inst.write('SENSe1:AVERage:MODE ' + mode)
        self.inst.write('SENSe1:AVERage:COUNt ' + str(num))
        self.inst.write('SENSe1:AVERage:STATe 1')

    def set_bandwidth(self, bandwidth: float):
        """Set the IF bandwidth

        Args:
            bandwidth (float): IF bandwidth in Hz
        """
        self.inst.write('SENS1:BAND ' + str(bandwidth))  # IFBW 1kHz

    def set_num_points(self, numpoints: int):
        """Set the number of sweep points

        Args:
            numpoints (int): Number of sweep points
        """
        self.inst.write('SENS1:SWE:POIN ' + str(numpoints))
        print(str(numpoints))

    # Create a measurement for the specified parameter
    def create_measurement(self, mname, param):
        """Create a measurement setup

        Args:
            mname (str): Measurement name
            param (str): Parameter name ('S21', 'S12', etc.)
        """
        self.inst.write('CALCulate1:PARameter:DEFine:EXT "' + mname + '",' + param)
        self.inst.write('DISPlay:WINDow1:TRACe1:FEED "' + mname + '"')
        self.inst.write('CALC1:FORM MLOG')  # Change the format to MLINear, MLOG, PHASe, etc.

    def run_sweep(self):
        """Run a single sweep and retrieve data

        Returns:
            list: List of data points from the sweep
        """
        OPC = self.inst.query('SENS:SWE:MODE SING;*OPC?')
        self.inst.write('CALC:PAR:MNUM 1')  # Selects trace number
        self.inst.write("CALC:DATA? SDATA")
        output = self.inst.read().split(',')
        return output

    def preset(self):
        """Display the sweep on the screen"""
        self.inst.write('SYST:FPR')
        self.inst.write('DISPlay:WINDow1:STATE ON')

    def set_freq(self, freq):
        """Set the center frequency

        Args:
            freq (float): Center frequency in GHz
        """
        self.inst.write(f'SENS:FREQ:CW {freq} GHz')
