
#import pyvisa as visa
#import time
from .LAN_instrument_base import Instrument

class ENA(Instrument):

    def set_power(self, power: float):
        self.inst.write('SOUR1:POW '+str(power))

    #frequency in GHz
    def set_freq_range(self, startfreq: str, stopfreq: str):
        self.inst.write('SENS1:FREQ:STAR '+startfreq+'E9')
        self.inst.write('SENS1:FREQ:STOP '+stopfreq+'E9')   
        
    #mode is either 'POINt' or 'SWEEP'
    def set_averaging(self, mode, num: int):
        self.inst.write('SENSe1:AVERage:MODE ' +mode)
        self.inst.write('SENSe1:AVERage:COUNt ' + str(num))
        self.inst.write('SENSe1:AVERage:STATe 1')
        
    def set_bandwidth(self, bandwidth: float):
        self.inst.write('SENS1:BAND '+str(bandwidth))          #IFBW 1kHz
        
    def set_num_points(self, numpoints: int):
        self.inst.write('SENS1:SWE:POIN '+str(numpoints))
        print(str(numpoints))
    
    #param is 'S21', 'S12', etc...
    def create_measurement(self, mname, param):
        
        self.inst.write('CALCulate1:PARameter:DEFine:EXT "'+mname +'",'+param)
        self.inst.write('DISPlay:WINDow1:TRACe1:FEED "' +mname +'"')
        self.inst.write('CALC1:FORM MLOG')                    #change the format  MLINear, MLOG,PHASe,UPH,IMAG,REAL,POLar,SMITh,SADM,SWR,GDEL,PPH,...
        
    def run_sweep(self):
        OPC=self.inst.query('SENS:SWE:MODE SING;*OPC?')
        #print(OPC)
        #self.inst.write('CALC1:MEAS1:DATA:SNP:PORTs:Save "1,2", "D:/Myfile.s2p"' );
        
        #self.inst.write('SENS:SWE:MODE CONT')
        self.inst.write('CALC:PAR:MNUM 1') # Selects trace number
        #print("Ran Okay")
        
        self.inst.write("CALC:DATA? SDATA")
        output =  self.inst.read().split(',')
        
        return output
        
    def preset(self):
        self.inst.write('SYST:FPR')
        self.inst.write('DISPlay:WINDow1:STATE ON')
    
    
    
    