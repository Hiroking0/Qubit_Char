# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 12:17:01 2022

@author: lqc
"""
import sys
sys.path.append("../")

from instruments.TekAwg import tek_awg as tawg
import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter.filedialog import askdirectory

#Returns awg var
def get_awg():
    ip = '172.20.1.5'
    port=5000

    awg = tawg.TekAwg.connect_raw_visa_socket(ip, port)
    awg.write("*CLS")
    #maybe use awg.write("*RST")
    
    return awg

#turns channels on
#makes sure pattern repeat is set correctly
#TODO: makes sure wait and jumps are set correctly
def init_awg(awg, num_patterns, pattern_repeat):
    for i in range(num_patterns):
        #awg.set_seq_element_jmp_type(i+1, 'off')
        awg.set_seq_element_loop_cnt(i+1, pattern_repeat)

    awg.set_chan_state(1, channel = [1,2])
    

#It should have show, make functions



class Wave_Arrs():
    def __init__(self, shape):
        self.c1 = np.zeros(shape, dtype = np.float32)
        self.c1m1 = np.zeros(shape, dtype = np.float32)
        self.c1m2 = np.zeros(shape, dtype = np.float32)
        self.c2 = np.zeros(shape, dtype = np.float32)
        self.c3 = np.zeros(shape, dtype = np.float32)
        self.c4 = np.zeros(shape, dtype = np.float32)
        self.c2m1 = np.zeros(shape, dtype = np.float32)
        self.c2m2 = np.zeros(shape, dtype = np.float32)
        
    def get_return_vals(self):
        return (self.c1, self.c1m1, self.c2, self.c2m1, self.c2m2, self.c3, self.c4)


class Pulse:
    #I think for now, assume everything is in samples
    #duration is length of high signal. So the signal will be
    #outputting value 'amplitude' from start to start+duration
    #markers should be numpy arrays or integer (0,1)
    #try using m1 = [start, duration, amplitude]

    def __init__(self, start, duration, amplitude, channel):

        self.duration = int(duration)
        self.start = int(start)
        self.amplitude = amplitude
        self.channel = channel

    def is_readout(self):
        return False

    def show(self):
        c1, c1m1, c1m2, c2, c2m1, c2m2 = self.make()
        plt.subplot(2,2,1)
        plt.plot(c1)
        plt.title("channel 1")
        
        plt.subplot(2,2,2)
        plt.plot(c2m1)
        plt.title("c2m1")

        plt.subplot(2,2,3)
        plt.plot(c2m2)
        plt.title("c2m2")

        plt.subplot(2,2,4)
        plt.plot(c1m1)
        plt.title("c1m1")
        
        plt.show()

    #for pulse return array of correct 
    def make(self, pad_length = None):
        if pad_length == None:
            length = self.start + self.duration
        else:
            length = pad_length
        final_arr = np.zeros(length, dtype = np.float32)
        c1m1 = np.zeros(length, dtype = np.float32)
        
        wave_set = Wave_Arrs(length)

        if self.channel == 1:
            final_arr[self.start : self.duration + self.start] += self.amplitude
        
        c1m1[self.start - PulseGroup.RF_TRIGGER_OFFSET : self.start - PulseGroup.RF_TRIGGER_OFFSET + PulseGroup.RF_TRIGGER_LENGTH] += 1
        setattr(wave_set, 'c'+str(self.channel), final_arr)
        wave_set.c1m1 = c1m1

        return wave_set.get_return_vals()
    
    
class gaussian():
    def __init__(self, mu, amplitude: float, gap: int,sigma: float,freq: float,numsig, channel: int):
        self.mu = mu
        self.amplitude = amplitude
        self.gap = gap
        self.sigma = sigma
        self.channel = channel
        self.freq = freq
        self.numsig = numsig


    def make(self, pad_length = None):
        if pad_length == None:
            length = self.mu + self.sigma*self.numsig+self.gap
        else:
            length = pad_length
        
        def gaussian(x, mu, sig):
            normalization = np.max(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))
            return self.amplitude*0.5/normalization*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        def cos(x,freq,shift):
            return np.cos(2*np.pi*x*freq + shift)
        
        final_arr = np.zeros(length, dtype = np.float32)
        
        time_array = np.linspace(self.mu-self.sigma*self.numsig, self.mu+self.sigma*self.numsig , int(self.sigma*self.numsig*2))
        cos_arr1 = gaussian(time_array, self.mu, self.sigma)*cos(time_array,self.freq,0)
        final_arr[int(self.mu-self.sigma*self.numsig): int(self.mu+self.sigma*self.numsig)] = cos_arr1
        #print(self.freq*1e-9)
        
        wave_set = Wave_Arrs(length)
        
        setattr(wave_set, 'c'+str(self.channel), final_arr)
        return wave_set.get_return_vals()
        

    

#readout pulse will be on c2m1
#readout trigger for alazar on c2m2
class Readout_Pulse(Pulse):

    def __init__(self, start, duration, amplitude):
        super().__init__(start, duration, amplitude, channel = None)

    def is_readout(self):
        return True

    def make(self, t_len = None):
        length = self.start + self.duration# + self.wait_time
        
        wave_set = Wave_Arrs(length)
        
        c2m1 = np.zeros(length, dtype = np.float32)
        c2m2 = np.zeros(length, dtype = np.float32)
        
        c2m1[self.start : self.start + self.duration] += self.amplitude
        c2m2[self.start - PulseGroup.READOUT_TRIGGER_OFFSET : self.start - PulseGroup.READOUT_TRIGGER_OFFSET + PulseGroup.READOUT_TRIGGER_LENGTH] += 1
    
        setattr(wave_set, 'c2m1', c2m1)
        setattr(wave_set, 'c2m2', c2m2)
        
        
        return wave_set.get_return_vals()


class Sweep_Pulse(Pulse):
    def __init__(self, start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step):
        super().__init__(start, duration, amplitude, channel)
        self.sweep_stop = sweep_stop
        self.sweep_step = sweep_step
        self.frequency = frequency
        self.phase = phase
    def make(self):
        raise NotImplementedError("Make function not implemented")
        
        
        
class Amp_Sweep_Gaussian(Sweep_Pulse):
    def __init__(self, start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step, total_sigma = 6):
        super().__init__(start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step)
        self.numsig = total_sigma #This is the number of sigma that the array will be zero outside of
        
    def make(self, length = 0):
        sweeps = np.arange(self.amplitude, self.sweep_stop, self.sweep_step)
        num_sweeps = len(sweeps)
        longest_length = max(length,self.start + self.duration)
        shape = (num_sweeps, int(longest_length))
        final_arr_1 = np.zeros(shape, dtype = np.float32)
        
        
        def gaussian(x, mu, sig,amp):
            return amp*0.5*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
        def cos(x,freq,shift):
            return np.cos(2*np.pi*x*freq + shift)
        
        time_array = np.linspace(self.start ,self.start + self.duration , int(self.duration))
        time_array2 = np.linspace(0,self.duration , int(self.duration))
        sigma = self.duration/(2*self.numsig)
        

        for ind, amp in enumerate(sweeps):
            cos_arr1 = gaussian(time_array, self.start + self.duration/2, sigma,amp)*cos(time_array2,self.frequency,0)
            final_arr_1[ind][self.start: self.start + self.duration ] = cos_arr1
        wave_set = Wave_Arrs(shape)
        
        setattr(wave_set, 'c'+str(self.channel), final_arr_1)
        return wave_set.get_return_vals()

        
        
class Duration_Sweep_Gaussian(Sweep_Pulse):
    def __init__(self, start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step, total_sigma = 6):
        super().__init__(start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step)
        self.numsig = total_sigma #This is the number of sigma that the array will be zero outside of
        
    def make(self, length = 0):
        sweeps = np.arange(self.duration, self.sweep_stop, self.sweep_step)
        num_sweeps = len(sweeps)
        longest_length = max(length, self.duration + self.start)
        shape = (num_sweeps, int(longest_length))
        # should be the max gap or sigma to make time_array and time2
        final_arr_1 = np.zeros(shape, dtype = np.float32)
            
        def gaussian(x, mu, sig):
            return self.amplitude*0.5*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        
        def cos(x,freq,shift):
            return np.cos(2*np.pi*x*freq + shift)
            
            
        #create the 2d array
        for ind, duration in enumerate(sweeps):

            mu = duration/2
            sigma = duration/(2*self.numsig)
            endpoint = self.start + self.duration
            startpoint = endpoint - duration

            time_array = np.linspace(0,duration , int(duration))
            time_array2 = np.linspace(0,duration , int(duration))
            cos_arr1 = gaussian(time_array, mu, sigma)*cos(time_array2, self.frequency, self.phase)
            final_arr_1[ind][int(startpoint): int(endpoint) ] = cos_arr1
            
            
        wave_set = Wave_Arrs(shape)
        setattr(wave_set, 'c'+str(self.channel), final_arr_1)
        
        return wave_set.get_return_vals()
    
class gap_sweep_gaussian(Sweep_Pulse):
    def __init__(self, initial_startpoint, duration, amplitude, freq, phase, channel, sweep_stop, sweep_step, total_sigma):
        super().__init__(initial_startpoint, duration, amplitude, freq, phase, channel, sweep_stop, sweep_step)
        self.numsig = total_sigma/2 #This is the number of sigma that the array will be zero outside of
    def make(self, length = 0):
        sweeps = np.arange(self.sweep_stop,self.start, self.sweep_step)
        num_sweeps = len(sweeps)
        longest_length = max(length, self.start + self.duration)
        shape = (num_sweeps, longest_length)
        final_arr_1 = np.zeros((num_sweeps, longest_length), dtype = np.float32)
        
        def gaussian(x, mu, sig):
            return self.amplitude*0.5*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        
        def cos(x,freq,shift):
            return np.cos(2*np.pi*x*freq + shift)
        time_array = np.linspace(0,self.duration, int(self.duration))
        time_array2 = np.linspace(0,self.duration , int(self.duration))
        mu = self.duration/2
        sigma = self.duration/(2*self.numsig)

        for ind, startpoint in enumerate(sweeps):
            cos_arr1 = gaussian(time_array, mu,sigma)*cos(time_array2,self.frequency,self.phase)
            final_arr_1[ind][int(startpoint): int(startpoint + self.duration) ] = cos_arr1
            
        final_arr_1 = final_arr_1[::-1]
            
        wave_set = Wave_Arrs(shape)
        setattr(wave_set, 'c'+str(self.channel), final_arr_1)
        
        return wave_set.get_return_vals()



        
class Amp_Sweep_Pulse(Sweep_Pulse):
    def __init__(self, start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step):
        super().__init__(start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step)
        
    def make(self, length = 0):
        sweeps = np.arange(self.amplitude, self.sweep_stop, self.sweep_step)
        num_sweeps = len(sweeps)
        longest_length = max(length, self.start + self.duration)
        shape = (num_sweeps, longest_length)
        final_arr_1 = np.zeros(shape, dtype = np.float32)
        for ind, amp in enumerate(sweeps):
            cos_arr1 = [amp*np.cos((self.frequency)*np.pi*2*i + self.phase) for i in range(self.duration)]
            final_arr_1[ind][self.start : self.start + self.duration] = cos_arr1
            
        wave_set = Wave_Arrs(shape)
        setattr(wave_set, 'c'+str(self.channel), final_arr_1)
        
        return wave_set.get_return_vals()
            
        
class Start_Sweep_Pulse(Sweep_Pulse):
    def __init__(self, start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step):
        super().__init__(start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step)
        
    def make(self, length = 0):
        sweeps = np.arange(self.start, self.sweep_stop, self.sweep_step)
        num_sweeps = len(sweeps)
        longest_length = max(length, max(sweeps) + self.duration)
        shape = (num_sweeps, longest_length)
        final_arr_1 = np.zeros(shape, dtype = np.float32)
        
        cos_arr1 = [self.amplitude*np.cos((self.frequency)*np.pi*2*i + self.phase) for i in range(self.duration)]
        for ind, start in enumerate(sweeps):
            final_arr_1[ind][start : start + self.duration] = cos_arr1
            
        #reverse array
        final_arr_1 = final_arr_1[::-1]
        
        wave_set = Wave_Arrs(shape)
        setattr(wave_set, 'c'+str(self.channel), final_arr_1)
        
        return wave_set.get_return_vals()
    
    
class Duration_Sweep_Pulse(Sweep_Pulse):
    def __init__(self, start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step):
        super().__init__(start, duration, amplitude, frequency, phase, channel, sweep_stop, sweep_step)
        
    def make(self, length = 0):
        sweeps = np.arange(self.duration, self.sweep_stop, self.sweep_step)
        num_sweeps = len(sweeps)
        longest_length = max(length, self.start + self.duration)
        shape = (num_sweeps, longest_length)
        final_arr_1 = np.zeros(shape, dtype = np.float32)
        
        #Create the longest arrays, then slice them to get the correct length
        t_cos_arr1 = [self.amplitude*np.cos((self.frequency)*np.pi*2*i + self.phase) for i in range(self.sweep_stop)]
        for ind, duration in enumerate(sweeps):
            final_arr_1[ind][self.start + self.duration - duration : self.start + self.duration] = t_cos_arr1[:duration]

        wave_set = Wave_Arrs(shape)
        setattr(wave_set, 'c'+str(self.channel), final_arr_1)
        
        return wave_set.get_return_vals()


class Sin_Pulse(Pulse):
    
    def __init__(self, start, duration, amplitude, frequency, phase, channel):
        super().__init__(start, duration, amplitude, channel)
        self.frequency = frequency*1e9
        self.phase = phase
        
    def make(self, pad_length = None):
        if pad_length == None:
            length = self.start + self.duration
        else:
            length = pad_length

        
        final_arr_1 = [self.amplitude*np.cos((self.frequency)*np.pi*2*i + self.phase) for i in range(self.duration)]
        
        wave_set = Wave_Arrs(shape)
        setattr(wave_set, 'c'+str(self.channel), final_arr_1)

        return wave_set.get_return_vals()
        

#TODO: make sweep_pulse class
#make "make" function for each of those classes that return 2d array [pattern#][pulse]
        
class PulseGroup:
        
    RF_TRIGGER_OFFSET = 100
    RF_TRIGGER_LENGTH = 100
    READOUT_TRIGGER_OFFSET = 800 #how long before readout should alazar trigger happen in nS
    READOUT_TRIGGER_LENGTH = 50

    def __init__(self, pulses = []):
        self.pulses = pulses
        

    def make(self):
        num_sweeps = 1
        total_length = None
        #go through all pulses and find the readout to see how 
        for pulse in self.pulses:
            if isinstance(pulse, Sweep_Pulse):
                sweep_made = pulse.make()
                num_sweeps = len(sweep_made[0])
                #break saves some computation, not necessary
            elif isinstance(pulse, Readout_Pulse):
                total_length = pulse.start + pulse.duration# + pulse.wait_time

        if total_length == None:
            print("NEED READOUT PULSE")
            return
        
        c1 = np.zeros((num_sweeps, total_length), dtype = np.float32)
        c1m1 = np.zeros((num_sweeps, total_length), dtype = np.float32)
        #c1m2 = np.zeros((num_sweeps, total_length), dtype = np.float32)
        c2 = np.zeros((num_sweeps, total_length), dtype = np.float32)
        c2m1 = np.zeros((num_sweeps, total_length), dtype = np.float32)
        c2m2 = np.zeros((num_sweeps, total_length), dtype = np.float32)
        c3 = np.zeros((num_sweeps, total_length), dtype = np.float32)
        c4 = np.zeros((num_sweeps, total_length), dtype = np.float32)
        for pulse in self.pulses:
            (t1, t1m1, t2, t2m1, t2m2, t3, t4) = pulse.make(total_length)
            for ind in range(num_sweeps):
                #add each pulse to its corresponding array
                
                if t1.ndim > 1:
                    c1[ind] = np.add(c1[ind], t1[ind])
                    c1m1[ind] = np.add(c1m1[ind], t1m1[ind])
                    c2[ind] = np.add(c2[ind], t2[ind])
                    c2m1[ind] = np.add(c2m1[ind], t2m1[ind])
                    c2m2[ind] = np.add(c2m2[ind], t2m2[ind])
                    c3[ind] = np.add(c3[ind], t3[ind])
                    c4[ind] = np.add(c4[ind], t4[ind])
                    
                else:
                    c1[ind] = np.add(c1[ind], t1)
                    c1m1[ind] = np.add(c1m1[ind], t1m1)
                    c2[ind] = np.add(c2[ind], t2)
                    c2m1[ind] = np.add(c2m1[ind], t2m1)
                    c2m2[ind] = np.add(c2m2[ind], t2m2)
                    c3[ind] = np.add(c3[ind], t3)
                    c4[ind] = np.add(c4[ind], t4)
                    

        #make all of the pulses in the array.
        #if its a sweep, it will be shape (num_sweeps, length_n)
        #if its not a sweep, it will be just shape (length_n)
        #add all makes together after reshaping them to have the same length.
        
        return (c1, c1m1, c2, c2m1, c2m2, c3, c4)

    def get_waveforms(self):
        
        
        (c1, c1m1, c2, c2m1, c2m2, c3, c4) = self.make()
        c1Waves = []
        c2Waves = []
        c3Waves = []
        c4Waves = []
        if c1.ndim == 1:
            c1Waves.append(tawg.Waveform(c1, c1m1, 0))
            c2Waves.append(tawg.Waveform(c2, c2m1, 0))
            c3Waves.append(tawg.Waveform(c3, 0, 0))
            c4Waves.append(tawg.Waveform(c4, 0, 0))
        else:
            for i in range(len(c1)):
                c1Waves.append(tawg.Waveform(c1[i], c1m1[i], 0))
                c2Waves.append(tawg.Waveform(c2[i], c2m1[i], c2m2[i]))
                c3Waves.append(tawg.Waveform(c3[i], 0, 0))
                c4Waves.append(tawg.Waveform(c4[i], 0, 0))
        
        return (c1Waves, c2Waves, c3Waves, c4Waves)

    #this function will send waves and sequence them
    def send_waves_awg(self, awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset, num_channels, decimation):
        PulseGroup.READOUT_TRIGGER_OFFSET = int(readout_trigger_offset/decimation)
        zero_multiple = int(zero_multiple/decimation)
        #First convert all arrays to waveforms, then send
        (c1Waves, c2Waves, c3Waves, c4Waves) = self.get_waveforms()
        
        awg.delete_all_waveforms()
        awg.delete_all_subseq()
        time.sleep(.5)
        
        send_waves(awg, c1Waves, name, channel = 1)
        time.sleep(.1)
        send_waves(awg, c2Waves, name, channel = 2)
        time.sleep(.1)
        if num_channels == 4:
            send_waves(awg, c3Waves, name, channel = 3)
            time.sleep(.1)
            send_waves(awg, c4Waves, name, channel = 4)
        
        awg.new_waveform('zero', tawg.Waveform(np.array([0]*zero_length, dtype = np.float32), 0, 0))
        
        time.sleep(.1)
        subseq_waves(awg, c1Waves, name, pattern_repeat, zero_multiple, num_channels)
        



    #This function will add one column of a pulse. All params should be Pulse objects
    def add_pulse(self, pulse):
        self.pulses.append(pulse)
    
    
    def show(self, decimation = 1, subplots = True):
        PulseGroup.READOUT_TRIGGER_OFFSET = int(PulseGroup.READOUT_TRIGGER_OFFSET/decimation)
        (arr1, a1m1, arr2, a2m1, a2m2, a3, a4) = self.make()
        x = np.linspace(0, len(arr1[0])*decimation, num = len(arr1[0]), endpoint=False)
        if arr1.ndim == 1:
            arr1 = np.expand_dims(arr1, axis=0)
            a1m1 = np.expand_dims(a1m1, axis=0)
            arr2 = np.expand_dims(arr2, axis=0)
            a2m1 = np.expand_dims(a2m1, axis=0)
            a2m2 = np.expand_dims(a2m2, axis=0)
            
        if subplots:
            step = 2
            for i in range(len(arr1)):
                plt.subplot(2,4,1)
                plt.plot(x, arr1[i]+step*i)
                plt.title('CH1 (qubit pulse)', fontsize=14)
                plt.xticks(rotation=45)

                plt.subplot(2,4,2)
                plt.plot(x, arr2[i]+step*i)
                plt.title('CH2 (qubit pulse)', fontsize=14)
                plt.xticks(rotation=45)

                plt.subplot(2,4,3)
                plt.plot(x, a1m1[i]+step*i)
                plt.title('CH1M1 (aux pulse unused)', fontsize=14)
                plt.xticks(rotation=45)

                plt.subplot(2,4,4)
                plt.plot(x, a2m2[i]+step*i)
                plt.title('CH2M2 (alazar trigger)', fontsize=14)
                plt.xticks(rotation=45)                

                plt.subplot(2,4,5)
                plt.plot(x, a3[i]+step*i)
                plt.title('CH3 (SSB Readout)', fontsize=14)
                plt.xticks(rotation=45)
                
                plt.subplot(2,4,6)
                plt.plot(x, a4[i]+step*i)
                plt.title('CH4 (SSB Readout)', fontsize=14)
                plt.xticks(rotation=45)
                
                plt.subplot(2,4,7)
                plt.plot(x, a2m1[i]+step*i)
                plt.title('CH2M1 (DC readout)', fontsize=14)
                plt.xticks(rotation=45)
                
                #plt.subplot(2,3,6)
                #plt.plot(a2m2[i])

        else:
            for i in range(len(arr1)):
                plt.plot(x, arr1[i]+i*1.1, 'b')
                #plt.plot(a1m1[0])
                #plt.plot(a1m2[0])
                #plt.plot(arr2[0])
                plt.plot(x, a2m1[i]+i*1.1, 'r')
                plt.plot(x, arr2[i]+i*1.1, 'g')
                #plt.plot(a2m2[i])
                #plt.legend(["ch1", "ch2m1", "ch2"])
                plt.xticks(rotation=45)
                #plt.tight_layout()
                
        plt.show()
        
        
    def to_file(self, name):
        directory = askdirectory(title='Select Folder')
        c1, c1m1, c1m2, c2, c2m1, c2m2 = self.make()
        f_name = directory + '/' + name
        for n_pattern in range(len(c1)):
            #save file for ch1 and ch2 name_channel_pattern
            with open(f_name + '_1_'+str(n_pattern) + ".txt", 'a') as f:
                print(f_name + '_1_'+str(n_pattern))
                for i in range(len(c1[0])):
                    f.write(str(c1[n_pattern][i]) + ',' + str(c1m1[n_pattern][i]) + ',' + str(c1m2[n_pattern][i]) + '\n')
            with open(f_name + '_2_'+str(n_pattern) + ".txt", 'a') as f:
                for i in range(len(c2[0])):
                    f.write(str(c2[n_pattern][i]) + ',' + str(c2m1[n_pattern][i]) + ',' + str(c2m2[n_pattern][i]) + '\n')
                

        
#this takes an array of Waveforms, sends them to awg with name format
#name_0, name_1... name_n-1
def send_waves(awg, arr, name, channel = 1):
    i = 0
    for wave in arr:
        t_name = name + "_" + str(channel) + "_" +str(i)
        print(t_name)
        i += 1
        awg.new_waveform(t_name, wave)
    
    
    
"""This function is used for waves with long wait times"""
#wait time in us
def subseq_waves(awg, arr, name, pattern_repeat, zero_repeat, num_channels):
    zero_repeat = max(zero_repeat, 1)
    awg.set_run_mode('seq')
    w_len = len(arr)
    awg.set_seq_length(w_len)
    #create a subsequence for each wave (ch1 ch2) and zero pair
    #first entry should have wait true
    
    #first create all the subequences
    for i in range(w_len):
        print("subsequencing: " + str(i))
        #create new subseq called ex "rabi_0... rabi_i"
        subseq_name = name + "_" + str(i)
        awg.new_subseq(subseq_name, 2)
        for j in range(num_channels):
            t_name = name + f"_{j+1}_" + str(i)
            #set each subsequence element. Channel 1 and 2
            awg.set_subseq_element(subseq_name, t_name, 1, j+1)
            awg.set_subseq_element(subseq_name, "zero", 2, j+1)
            #Set each repeat of zero to correct number
            awg.set_subseq_repeat(subseq_name, 2, zero_repeat)
            
            
        #finally set the main sequence to have the subsequence entry
        awg.set_seq_elm_to_subseq(i+1, subseq_name)
        awg.set_seq_element_loop_cnt(i+1, pattern_repeat)
    
def seq_waves(awg, arr, name, pattern_repeat):
    
    
    awg.set_run_mode('seq')
    w_len = len(arr)
    awg.set_seq_length(w_len)
    #first entry should have wait true
    t_name1 = name + "_1_0"
    t_name2 = name + "_2_0"
    #print(t_name1, t_name2)
    t_ent = tawg.SequenceEntry([t_name1, t_name2, None, None], wait = True, loop_count = pattern_repeat)
    awg.set_seq_element(1,t_ent)
    
    for i in range(1, w_len-1):
        t_name1 = name + "_1_" + str(i)
        t_name2 = name + "_2_" + str(i)
        #print("in loop", t_name1, t_name2)
        t_ent = tawg.SequenceEntry([t_name1, t_name2, None, None], loop_count = pattern_repeat)
        
        awg.set_seq_element(i+1, t_ent)
        time.sleep(.01)
    
    #last entry should have goto 1
    
    t_name1 = name + "_1_" + str(w_len-1)
    t_name2 = name + "_2_" + str(w_len-1)
    #print("out of loop", t_name1, t_name2)
    t_ent = tawg.SequenceEntry([t_name1, t_name2, None, None], loop_count = pattern_repeat, goto_state = 'ind', goto_ind = 1)
    awg.set_seq_element(w_len,t_ent)
    
if __name__ == "__main__":

    name = "here"
            #start, duration, amplitude, channel, sweep_type, sweep_end, sweep_step, readout, wait_time
    #pg.to_file("hi")
    #(self, start, duration, amplitude, channel):
    p1 = Pulse(1000, 2000, 1, 1)
    #m = p1.make(200)
    #p1.show()

    #(self, start, duration, amplitude, sweep_param, sweep_stop, sweep_step, channel = None):
    p2 = Sweep_Pulse(3200, 100, 1, 'duration', 200, 10, 1)

    #(self, start, duration, amplitude, wait_time):
    ro = Readout_Pulse(5000, 1000, 1, 100)
    #ro.show()

    pg = PulseGroup([p1, p2, ro])
    #pg.show()
    pg.to_file("t_seq")

