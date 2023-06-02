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
        c1 = np.zeros(length, dtype = np.float32)
        c1m1 = np.zeros(length, dtype = np.float32)
        c1m2 = np.zeros(length, dtype = np.float32)
        c2 = np.zeros(length, dtype = np.float32)
        c2m1 = np.zeros(length, dtype = np.float32)
        c2m2 = np.zeros(length, dtype = np.float32)

        if self.channel == 1:
            c1[self.start : self.duration + self.start] += self.amplitude
        else:
            c2[self.start : self.duration + self.start] += self.amplitude
        
        #if self.trigger_before_pulse:
        #    c1m1[self.start - PulseGroup.RF_TRIGGER_OFFSET : self.start - PulseGroup.RF_TRIGGER_OFFSET + RF_TRIGGER_LENGTH] += 1          
        c1m1[self.start - PulseGroup.RF_TRIGGER_OFFSET : self.start - PulseGroup.RF_TRIGGER_OFFSET + PulseGroup.RF_TRIGGER_LENGTH] += 1

        return (c1,c1m1,c1m2,c2,c2m1,c2m2)

#readout pulse will be on c2m1
#readout trigger for alazar on c2m2
class Readout_Pulse(Pulse):

    def __init__(self, start, duration, amplitude, wait_time = 777):
        super().__init__(start, duration, amplitude, channel = None)
        self.wait_time = wait_time

    def is_readout(self):
        return True

    def make(self, t_len = 0):
        length = self.start + self.duration# + self.wait_time
        c1 = np.zeros(length, dtype = np.float32)
        c1m1 = np.zeros(length, dtype = np.float32)
        c1m2 = np.zeros(length, dtype = np.float32)
        c2 = np.zeros(length, dtype = np.float32)

        c2m1 = np.zeros(length, dtype = np.float32)
        c2m1[self.start : self.start + self.duration] += self.amplitude

        c2m2 = np.zeros(length, dtype = np.float32)
        c2m2[self.start - PulseGroup.READOUT_TRIGGER_OFFSET : self.start - PulseGroup.READOUT_TRIGGER_OFFSET + PulseGroup.READOUT_TRIGGER_LENGTH] += 1


        return (c1,c1m1,c1m2,c2,c2m1,c2m2)


class Sweep_Pulse(Pulse):
    #sweep_type can be 'amplitude' 'amp', 'duration' 'dur', or 'start'
    def __init__(self, start, duration, amplitude, sweep_param, sweep_stop, sweep_step, channel = None):
        super().__init__(start, duration, amplitude, channel)
        self.sweep_type = sweep_param
        self.sweep_stop = sweep_stop
        self.sweep_step = sweep_step

    #assume sweep_stop is NON-inclusive
    def make(self, length = 0):

        if self.sweep_type == "duration":
            
            sweeps = np.arange(self.duration, self.sweep_stop, self.sweep_step)
            num_sweeps = len(sweeps)
            longest_length = max(length, self.start + self.duration)
            final_arr = np.zeros((num_sweeps, longest_length), dtype = np.float32)
            for ind, duration in enumerate(sweeps):
                #self.start + self.duration - duration : self.start + self.duration
                final_arr[ind][self.start + self.duration - duration : self.start + self.duration] += self.amplitude

        elif self.sweep_type == "start":
            sweeps = np.arange(self.start, self.sweep_stop, self.sweep_step)
            num_sweeps = len(sweeps)
            longest_length = max(length, max(sweeps) + self.duration)
            
            
            final_arr = np.zeros((num_sweeps, longest_length), dtype = np.float32)
            for ind, start in enumerate(sweeps):
                final_arr[ind][start : start + self.duration] += self.amplitude
            final_arr = final_arr[::-1]
                
        elif self.sweep_type == "amplitude" or self.sweep_type == 'amp':
            sweeps = np.arange(self.amplitude, self.sweep_stop, self.sweep_step)
            num_sweeps = len(sweeps)
            longest_length = max(length, self.start + self.duration)

            final_arr = np.zeros((num_sweeps, longest_length), dtype = np.float32)
            for ind, amp in enumerate(sweeps):
                final_arr[ind][self.start : self.start + self.duration] += amp


        if self.channel == None or self.channel == 1:
            c1 = final_arr
            c2 = np.zeros((num_sweeps, longest_length), dtype = np.float32)
        else:
            c2 = final_arr
            c1 = np.zeros((num_sweeps, longest_length), dtype = np.float32)

        c1m1 = np.zeros((num_sweeps, longest_length), dtype = np.float32)
        c1m2 = np.zeros((num_sweeps, longest_length), dtype = np.float32)
        c2m1 = np.zeros((num_sweeps, longest_length), dtype = np.float32)
        c2m2 = np.zeros((num_sweeps, longest_length), dtype = np.float32)


        return c1, c1m1, c1m2, c2, c2m1, c2m2


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
        c1m2 = np.zeros((num_sweeps, total_length), dtype = np.float32)
        c2 = np.zeros((num_sweeps, total_length), dtype = np.float32)
        c2m1 = np.zeros((num_sweeps, total_length), dtype = np.float32)
        c2m2 = np.zeros((num_sweeps, total_length), dtype = np.float32)
        for ind in range(num_sweeps):
            for pulse in self.pulses:
                #add each pulse to its corresponding array
                (t1, t1m1, t1m2, t2, t2m1, t2m2) = pulse.make(total_length)
                if t1.ndim > 1:
                    c1[ind] = np.add(c1[ind], t1[ind])
                    c1m1[ind] = np.add(c1m1[ind], t1m1[ind])
                    c1m2[ind] = np.add(c1m2[ind], t1m2[ind])
                    c2[ind] = np.add(c2[ind], t2[ind])
                    c2m1[ind] = np.add(c2m1[ind], t2m1[ind])
                    c2m2[ind] = np.add(c2m2[ind], t2m2[ind])
                else:
                    c1[ind] = np.add(c1[ind], t1)
                    c1m1[ind] = np.add(c1m1[ind], t1m1)
                    c1m2[ind] = np.add(c1m2[ind], t1m2)
                    c2[ind] = np.add(c2[ind], t2)
                    c2m1[ind] = np.add(c2m1[ind], t2m1)
                    c2m2[ind] = np.add(c2m2[ind], t2m2)
        #make all of the pulses in the array.
        #if its a sweep, it will be shape (num_sweeps, length_n)
        #if its not a sweep, it will be just shape (length_n)
        #add all makes together after reshaping them to have the same length.
        
        return (c1, c1m1, c1m2, c2, c2m1, c2m2)

    def get_waveforms(self):
        (c1, c1m1, c1m2, c2, c2m1, c2m2) = self.make()
        c1Waves = []
        c2Waves = []
        if c1.ndim == 1:
            c1Waves.append(tawg.Waveform(c1, c1m1, c1m2))
            c2Waves.append(tawg.Waveform(c2, c2m1, c2m2))
        else:
            for i in range(len(c1)):
                c1Waves.append(tawg.Waveform(c1[i], c1m1[i], c1m2[i]))
                c2Waves.append(tawg.Waveform(c2[i], c2m1[i], c2m2[i]))
        
        return (c1Waves, c2Waves)

    #this function will send waves and sequence them
    def send_waves_awg(self, awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset, decimation):
        PulseGroup.READOUT_TRIGGER_OFFSET = int(readout_trigger_offset/decimation)
        zero_multiple = int(zero_multiple/decimation)
        #First convert all arrays to waveforms, then send
        (c1Waves, c2Waves) = self.get_waveforms()
        
        awg.delete_all_waveforms()
        awg.delete_all_subseq()
        time.sleep(.5)
        
        send_waves(awg, c1Waves, name, channel = 1)
        time.sleep(.1)
        send_waves(awg, c2Waves, name, channel = 2)
        
        awg.new_waveform('zero', tawg.Waveform(np.array([0]*zero_length, dtype = np.float32), 0, 0))
        
        time.sleep(.1)
        #seq_waves(awg, c1Waves, name, pattern_repeat)
        subseq_waves(awg, c1Waves, name, pattern_repeat, zero_multiple)
        



    #This function will add one column of a pulse. All params should be Pulse objects
    def add_pulse(self, pulse):
        self.pulses.append(pulse)
    
    
    def show(self, decimation = 1, subplots = True):
        PulseGroup.READOUT_TRIGGER_OFFSET = int(PulseGroup.READOUT_TRIGGER_OFFSET/decimation)
        (arr1, a1m1, a1m2, arr2, a2m1, a2m2) = self.make()
        x = np.linspace(0, len(arr1[0])*decimation, num = len(arr1[0]), endpoint=False)
        if arr1.ndim == 1:
            arr1 = np.expand_dims(arr1, axis=0)
            a1m1 = np.expand_dims(a1m1, axis=0)
            a1m2 = np.expand_dims(a1m2, axis=0)
            arr2 = np.expand_dims(arr2, axis=0)
            a2m1 = np.expand_dims(a2m1, axis=0)
            a2m2 = np.expand_dims(a2m2, axis=0)
            
        if subplots:
            for i in range(len(arr1)):
                plt.subplot(2,3,1)
                plt.plot(x, arr1[i]+i)
                plt.title('CH1 (qubit pulse)', fontsize=14)
                plt.xticks(rotation=45)

                plt.subplot(2,3,2)
                plt.plot(x, a1m1[i]+i)
                plt.title('CH1M1 (aux pulse)', fontsize=14)
                plt.xticks(rotation=45)

                plt.subplot(2,3,3)
                plt.plot(x, a1m2[i]+i)
                plt.title('CH1M2 (nothing)', fontsize=14)
                plt.xticks(rotation=45)

                plt.subplot(2,3,4)
                plt.plot(x, arr2[i]+i)
                plt.title('CH2 (qubit pulse)', fontsize=14)
                plt.xticks(rotation=45)                

                plt.subplot(2,3,5)
                plt.plot(x, a2m1[i]+i)
                plt.title('CH2 M1 (readout pulse)', fontsize=14)
                plt.xticks(rotation=45)
                
                plt.subplot(2,3,6)
                plt.plot(x, a2m2[i]+i)
                plt.title('CH2 M2 (Alazar trigger)', fontsize=14)
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
def subseq_waves(awg, arr, name, pattern_repeat, zero_repeat):
    
    #zero will be of length 500 us so the number it repeats should be integer division
    zero_repeat = max(zero_repeat, 1)
    awg.set_run_mode('seq')
    w_len = len(arr)
    awg.set_seq_length(w_len)
    
    #create a subsequence for each wave (ch1 ch2) and zero pair
    #first entry should have wait true
    
    #first create all the subequences
    for i in range(0, w_len):
        print("subsequencing: " + str(i))
        #create new subseq called ex "rabi_0... rabi_i"
        subseq_name = name + "_" + str(i)
        awg.new_subseq(subseq_name, 2)
        
        #set each subsequence element. Channel 1 and 2
        t_name1 = name + "_1_" + str(i)
        t_name2 = name + "_2_" + str(i)
        
        awg.set_subseq_element(subseq_name, t_name1, 1, 1)
        awg.set_subseq_element(subseq_name, t_name2, 1, 2)
        
        awg.set_subseq_element(subseq_name, "zero", 2, 1)
        awg.set_subseq_element(subseq_name, "zero", 2, 2)
        time.sleep(.2)
        
        #Set each repeat of zero to correct number
        awg.set_subseq_repeat(subseq_name, 2, zero_repeat)
        
        #finally set the main sequence to have the subsequence entry
        awg.set_seq_elm_to_subseq(i+1, subseq_name)
        awg.set_seq_element_loop_cnt(i+1, pattern_repeat)
        
    #for i in range(1, w_len):
    #    awg.set_seq_element_goto_state(i, 0)
    #awg.set_seq_element_goto_state(w_len, 1)
    #time.sleep(.2)
    #awg.set_seq_element_wait(1, 1)
        
    
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

