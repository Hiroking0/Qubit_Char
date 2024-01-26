# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:16:01 2023

@author: lqc
"""
import pyvisa
import numpy as np
import matplotlib.pyplot as plt

def getdata(start,stop,points,pow,bandwidth):
    rm = pyvisa.ResourceManager()
    inst = rm.open_resource('TCPIP0::K-E5080B-00202.local::hislip0::INSTR')
    ret = inst.query("*IDN?")
    inst.write('CALC:MEAS:PAR "S21"')

    #set params
    startfreq = start
    stopfreq = stop
    numpoints = points
    power = pow



    #set power
    inst.write('SOUR1:POW '+str(power))

    #set freq range
    inst.write('SENS1:FREQ:STAR '+ str(startfreq))
    inst.write('SENS1:FREQ:STOP '+str(stopfreq)) 

    #set number of points
    inst.write('SENS1:SWE:POIN '+str(numpoints))

    #set bandwidth
    inst.write('SENS:BWID ' + str(bandwidth)+'KHZ')

    #read data
    inst.write('CALC:PAR:MNUM 1')
    inst.write("CALC:DATA? SDATA")
    output =  inst.read().split(',')
    inst.close()

    #Magnitude real imaginary
    mag=np.zeros(int(len(output)/2))
    real=np.zeros(int(len(output)/2))
    Im=np.zeros(int(len(output)/2))
    for i in range(int(len(output)/2)):
        mag[i]=float(output[i*2])**2 + float(output[i*2+1])**2
        real[i]=float(output[i*2])
        Im[i]=float(output[i*2+1])


    #Plotting
    x=np.linspace(startfreq,stopfreq,len(mag))

    measured_s21 = real +1j*Im
    return measured_s21
def justgetdata():
    rm = pyvisa.ResourceManager()
    inst = rm.open_resource('TCPIP0::K-E5080B-00202.local::hislip0::INSTR')
    ret = inst.query("*IDN?")
    inst.write('CALC:MEAS:PAR "S21"')
    # Query the start frequency
    start_freq_str = inst.query('SENS1:FREQ:STAR?')
    start_freq = float(start_freq_str)

    # Query the stop frequency
    stop_freq_str = inst.query('SENS1:FREQ:STOP?')
    stop_freq = float(stop_freq_str)

    # Read data
    inst.write('CALC:PAR:MNUM 1')
    inst.write("CALC:DATA? SDATA")
    output = inst.read().split(",")

    # Magnitude, real, imaginary
    mag = np.zeros(int(len(output)/2))
    real = np.zeros(int(len(output)/2))
    Im = np.zeros(int(len(output)/2))
    for i in range(int(len(output)/2)):
        mag[i] = float(output[i*2])**2 + float(output[i*2+1])**2
        real[i] = float(output[i*2])
        Im[i] = float(output[i*2+1])

    # Plotting
    x = np.linspace(start_freq, stop_freq, len(mag))
    #print(start_freq, stop_freq)
    measured_s21 = real + 1j * Im

    # Close the instrument connection
    inst.close()
    return measured_s21,x*1e-9