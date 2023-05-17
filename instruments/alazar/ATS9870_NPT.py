from __future__ import division
import ctypes
import numpy as np
import os
#import signal
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'Library'))
from . import atsapi as ats
#import atsapi as ats
import matplotlib.pyplot as plt

samplesPerSec = None

# Configures a board for acquisition
def ConfigureBoard(board):
    # TODO: Select clock parameters as required to generate this
    # sample rate
    #
    # For example: if samplesPerSec is 100e6 (100 MS/s), then you can
    # either:
    #  - select clock source INTERNAL_CLOCK and sample rate
    #    SAMPLE_RATE_100MSPS
    #  - or select clock source FAST_EXTERNAL_CLOCK, sample rate
    #    SAMPLE_RATE_USER_DEF, and connect a 100MHz signal to the
    #    EXT CLK BNC connector
    global samplesPerSec
    samplesPerSec = 1000000000.0
    board.setCaptureClock(ats.EXTERNAL_CLOCK_10MHz_REF,
                          1000000000,
                          ats.CLOCK_EDGE_RISING,
                          1)
    #board.setCaptureClock(ats.INTERNAL_CLOCK,
    #                      ats.SAMPLE_RATE_1000MSPS,
    #                      ats.CLOCK_EDGE_RISING,
    #                      0)

    #board.setExternalClockLevel(50)
    
    #9870 only compatible with 400mv 200mv 100mv 40mv 1V
    sensitivity = ats.INPUT_RANGE_PM_100_MV
    
    # TODO: Select channel A input parameters as required.
    board.inputControl(ats.CHANNEL_A,
                         ats.DC_COUPLING,
                         sensitivity,
                         ats.IMPEDANCE_50_OHM)
    
    # TODO: Select channel A bandwidth limit as required.
    board.setBWLimit(ats.CHANNEL_A, 0)
    
    
    # TODO: Select channel B input parameters as required.
    board.inputControl(ats.CHANNEL_B,
                         ats.DC_COUPLING,
                         sensitivity,
                         ats.IMPEDANCE_50_OHM)
    
    # TODO: Select channel B bandwidth limit as required.
    board.setBWLimit(ats.CHANNEL_B, 0)
    
    # TODO: Select trigger inputs and levels as required.
    board.setTriggerOperation(ats.TRIG_ENGINE_OP_J,
                              ats.TRIG_ENGINE_J,
                              ats.TRIG_EXTERNAL,
                              ats.TRIGGER_SLOPE_POSITIVE,
                              150,
                              ats.TRIG_ENGINE_K,
                              ats.TRIG_DISABLE,
                              ats.TRIGGER_SLOPE_POSITIVE,
                              128)

    # TODO: Select external trigger parameters as required.
    board.setExternalTrigger(ats.DC_COUPLING,
                             ats.ETR_5V)

    # TODO: Set trigger delay as required.
    triggerDelay_sec = 0
    #triggerDelay_samples = int(triggerDelay_sec * samplesPerSec + 0.5)
    board.setTriggerDelay(0)

    # TODO: Set trigger timeout as required.
    #
    # NOTE: The board will wait for a for this amount of time for a
    # trigger event.  If a trigger event does not arrive, then the
    # board will automatically trigger. Set the trigger timeout value
    # to 0 to force the board to wait forever for a trigger event.
    #
    # IMPORTANT: The trigger timeout value should be set to zero after
    # appropriate trigger parameters have been determined, otherwise
    # the board may trigger if the timeout interval expires before a
    # hardware trigger event arrives.
    board.setTriggerTimeOut(0)

    # Configure AUX I/O connector as required
    #board.configureAuxIO(ats.AUX_OUT_TRIGGER,
    #                     0)
    

def AcquireData(board, params, num_patterns, path, saveData = True, live_plot = False):
    
    samp_per_acq = params['acq_multiples']*256
    pattern_repeat = params['pattern_repeat']
    seq_repeat = params['seq_repeat']
    avg_start = params['avg_start']
    avg_duration = params['avg_length']
    #print("samp per ac", samp_per_acq)
    #print("num patterns", num_patterns)
    #print("pattern repeat", pattern_repeat)
    #print("seq repeat", seq_repeat)
    #For our purposes, I believe the only params we should change are postTrigsamples
    #and buffs per acquisition
    #PTS should be a multiple of 256 that is close to the size of the acquisition length, in nS
    
    #chA_avgs = np.zeros(seq_repeat * pattern_repeat * num_patterns)
    #chB_avgs = np.zeros(seq_repeat * pattern_repeat * num_patterns)
    
    #dt = np.dtype(float, metadata=params)
    chA_avgs_sub = np.zeros((num_patterns, seq_repeat * pattern_repeat))
    chB_avgs_sub = np.zeros((num_patterns, seq_repeat * pattern_repeat))
    chA_avgs_nosub = np.zeros((num_patterns, seq_repeat * pattern_repeat))
    chB_avgs_nosub = np.zeros((num_patterns, seq_repeat * pattern_repeat))
    
    plt_avg = np.zeros(num_patterns)
    
    if live_plot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot(range(num_patterns), plt_avg) # Returns a tuple of line objects, thus the comma
    
    
    # No pre-trigger samples in NPT mode
    preTriggerSamples = 0

    #Select the number of samples per record.
    postTriggerSamples = samp_per_acq

    #Select the number of records per DMA buffer.
    recordsPerBuffer = 1

    #Select the number of buffers per acquisition.
    #in the NPT case, this will be the total number of acquisitions
    buffersPerAcquisition = num_patterns * pattern_repeat * seq_repeat
    #Select the active channels.
    channels = ats.CHANNEL_A | ats.CHANNEL_B
    channelCount = 0
    for c in ats.channels:
        channelCount += (c & channels == c)

    #Should data be saved to file?
    dataFile = None
    if saveData:
        
        dataFile = open(path + "rawdata.bin", 'wb')

    #if saveData:
    #    dataFileA = open(os.path.join(os.path.dirname(__file__),
    #                                 "dataA.bin"), 'wb')
    #    dataFileB = open(os.path.join(os.path.dirname(__file__),
    #                                 "dataB.bin"), 'wb')

    # Compute the number of bytes per record and per buffer
    memorySize_samples, bitsPerSample = board.getChannelInfo()
    bytesPerSample = (bitsPerSample.value + 7) // 8
    samplesPerRecord = preTriggerSamples + postTriggerSamples
    bytesPerRecord = bytesPerSample * samplesPerRecord
    bytesPerBuffer = bytesPerRecord * recordsPerBuffer * channelCount

    #Select number of DMA buffers to allocate
    bufferCount = 10

    # Allocate DMA buffers

    sample_type = ctypes.c_uint8
    if bytesPerSample > 1:
        sample_type = ctypes.c_uint16

    buffers = []
    for i in range(bufferCount):
        buffers.append(ats.DMABuffer(board.handle, sample_type, bytesPerBuffer))
    
    # Set the record size
    board.setRecordSize(preTriggerSamples, postTriggerSamples)

    recordsPerAcquisition = recordsPerBuffer * buffersPerAcquisition

    # Configure the board to make an NPT AutoDMA acquisition
    board.beforeAsyncRead(channels,
                          -preTriggerSamples,
                          samplesPerRecord,
                          recordsPerBuffer,
                          recordsPerAcquisition,
                          ats.ADMA_EXTERNAL_STARTCAPTURE | ats.ADMA_NPT)
    

    # Post DMA buffers to board
    for buffer in buffers:
        board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
    
    start = time.time() # Keep track of when acquisition started
    try:
        board.startCapture() # Start the acquisition
        print("Capturing %d buffers. Press <enter> to abort" %
              buffersPerAcquisition)
        buffersCompleted = 0
        bytesTransferred = 0
        while (buffersCompleted < buffersPerAcquisition and not
               ats.enter_pressed()):
            
            pattern_number = int(buffersCompleted/pattern_repeat) % num_patterns
            seq_number = int(buffersCompleted/(num_patterns*pattern_repeat))
            
            index_number = seq_number*pattern_repeat + buffersCompleted % pattern_repeat
            
            
            
            
            #print(pattern_number, index_number, seq_number, seq_number*pattern_repeat, buffersCompleted % pattern_repeat)
            
            #print(buffersCompleted)
            # Wait for the buffer at the head of the list of available
            # buffers to be filled by the board.
            buffer = buffers[buffersCompleted % len(buffers)]
            board.waitAsyncBufferComplete(buffer.addr, timeout_ms=5000)
            
            
            half = int(len(buffer.buffer)/2)
            chA = buffer.buffer[:half]
            chB = buffer.buffer[half:]
            
            t_Aavg = np.average(chA[avg_start: avg_start + avg_duration])
            t_Bavg = np.average(chB[avg_start: avg_start + avg_duration])
            
            
            chA_avgs_sub[pattern_number][index_number] = t_Aavg - np.average(chA[200:800])
            chB_avgs_sub[pattern_number][index_number] = t_Bavg - np.average(chB[200:800])
            chA_avgs_nosub[pattern_number][index_number] = t_Aavg
            chB_avgs_nosub[pattern_number][index_number] = t_Bavg
            
            
            
            
            
            
            if live_plot and pattern_number == 0:
                for i in range(num_patterns):
                    plt_avg[i] = np.average(chB_avgs_sub[i][:index_number])
                    #fig.canvas.title()
                
                
                line1.set_ydata(plt_avg)
                fig.canvas.draw()
                fig.canvas.flush_events()
                
            
            
            buffersCompleted += 1
            bytesTransferred += buffer.size_bytes
            
            
            # Process sample data in this buffer. Data is available
            # as a NumPy array at buffer.buffer

            # NOTE:
            #
            # While you are processing this buffer, the board is already
            # filling the next available buffer(s).
            #
            # You MUST finish processing this buffer and post it back to the
            # board before the board fills all of its available DMA buffers
            # and on-board memory.
            #
            # Samples are arranged in the buffer as follows:
            # S0A, S0B, ..., S1A, S1B, ...
            # with SXY the sample number X of channel Y.
            #
            # Sample code are stored as 8-bit values.
            #
            # Sample codes are unsigned by default. As a result:
            # - 0x00 represents a negative full scale input signal.
            # - 0x80 represents a ~0V signal.
            # - 0xFF represents a positive full scale input signal.
            # Optionaly save data to file
            if dataFile:
                buffer.buffer.tofile(dataFile)

            #if dataFile:
            #    half = int(len(buffer.buffer)/2)
            #    buffer.buffer[:half].tofile(dataFileA)
            #    buffer.buffer[half:].tofile(dataFileB)

            # Add the buffer to the end of the list of available buffers.
            board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
            
    finally:
        board.abortAsyncRead()
    # Compute the total transfer time, and display performance information.
    transferTime_sec = time.time() - start
    print("Capture completed in %f sec" % transferTime_sec)
    buffersPerSec = 0
    bytesPerSec = 0
    recordsPerSec = 0
    if transferTime_sec > 0:
        buffersPerSec = buffersCompleted / transferTime_sec
        bytesPerSec = bytesTransferred / transferTime_sec
        recordsPerSec = recordsPerBuffer * buffersCompleted / transferTime_sec
    print("Captured %d buffers (%f buffers per sec)" %
          (buffersCompleted, buffersPerSec))
    print("Captured %d records (%f records per sec)" %
          (recordsPerBuffer * buffersCompleted, recordsPerSec))
    print("Transferred %d bytes (%f bytes per sec)" %
          (bytesTransferred, bytesPerSec))
    #np.save(path + "chA_sub", chA_avgs_sub)
    #np.save(path + "chB_sub", chB_avgs_sub)
    #np.save(path + "chA_nosub", chA_avgs_nosub)
    #np.save(path + "chB_nosub", chB_avgs_nosub)
    mag_sub = np.zeros((len(chA_avgs_sub), len(chA_avgs_sub[0])))
    mag_nosub = np.zeros((len(chA_avgs_sub), len(chA_avgs_sub[0])))
    for i in range(len(chA_avgs_sub)):
        for j in range(len(chA_avgs_sub[0])):
            mag_sub[i][j] = np.sqrt(chA_avgs_sub[i][j] ** 2 + chB_avgs_sub[i][j] ** 2)
            mag_nosub[i][j] = np.sqrt(chA_avgs_nosub[i][j] ** 2 + chB_avgs_nosub[i][j] ** 2)
    
    #np.save(path + "mag_sub", mag_sub)
    #np.save(path + "mag_nosub", mag_nosub)
    
    if live_plot:
        print('closed')
        fig.canvas.close()
    
    return (chA_avgs_sub, chB_avgs_sub, chA_avgs_nosub, chB_avgs_nosub, mag_sub, mag_nosub)


if __name__ == "__main__":
    board = ats.Board(systemId = 1, boardId = 1)
    ConfigureBoard(board)
    AcquireData(board)
