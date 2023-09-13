from qcodes.instrument_drivers.tektronix import (
    TektronixAWG5014,  # <--- The instrument driver
)
from qcodes.instrument_drivers.tektronix.AWGFileParser import (
    parse_awg_file,  # <--- A helper function
)
from qcodes import VisaInstrument
import numpy as np
import os

# Create an instance of the Tektronix_AWG5014 class
awg1 = TektronixAWG5014('AWG1', 'TCPIP0::172.20.1.5::5000::SOCKET', timeout=40)

noofseqelems = 6
noofpoints = 1200
waveforms = [[], []]  # one list for each channel
m1s = [[], []]
m2s = [[], []]
for ii in range(noofseqelems):
    # waveform and markers for channel 1
    waveforms[0].append(np.sin(np.pi*(ii+1)*np.linspace(0, 1, noofpoints))*np.hanning(noofpoints))
    m1 = np.zeros(noofpoints)
    m1[:int(noofpoints/(ii+1))] = 1
    m1s[0].append(m1)
    m2 = np.zeros(noofpoints)
    m2s[0].append(m2)

    # waveform and markers for channel two
    wf = np.sin(np.pi*(ii+1)*np.linspace(0, 1, noofpoints))
    wf *= np.arctan(np.linspace(-20, 20, noofpoints))/np.pi*2
    waveforms[1].append(wf)
    m1 = np.zeros(noofpoints)
    m1[:int(noofpoints/(ii+1))] = 1
    m1s[1].append(m1)
    m2 = np.zeros(noofpoints)
    m2s[1].append(m2)

    # Sequencing options

# number of repetitions
nreps = [2 for ii in range(noofseqelems)]
# Wait trigger (0 or 1)
trig_waits = [0]*noofseqelems
# Goto state
goto_states = [((ii+1) % noofseqelems)+1 for ii in range(noofseqelems)]
#goto_states = [0]*noofseqelems
# Event jump
jump_tos = [2]*noofseqelems

filepath = os.path.join(os.getcwd(), 'test_awg_file.awg')
awgfile = awg1.make_and_save_awg_file(waveforms, m1s, m2s, nreps, trig_waits, goto_states,
                                      jump_tos, channels=[1, 3], filename=filepath)