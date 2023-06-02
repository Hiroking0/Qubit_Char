# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:09:10 2023

@author: lqc
"""

import sys
sys.path.append("../")

from lib import wave_construction as be
import matplotlib.pyplot as plt
import numpy as np

MHz = 1e6
GHz = 1e9


#self, start: int, duration: int, amplitude, frequency, channel: int, phase: float
sp1 = be.Sin_Pulse(10, 1000, 1, 100*MHz, 1, -np.pi/2)
ro = be.Readout_Pulse(2000, 20000, 1)

pg = be.PulseGroup([sp1, ro])
#c1, c1m1, c1m2, c2, c2m1, c2m2 = sp1.make()
#plt.plot(c1)
#plt.plot(c2)
#plt.figure()

#c1, c1m1, c1m2, c2, c2m1, c2m2 = sp2.make()
#plt.plot(c1)
#plt.plot(c2)
#plt.show()
pg.show()
#awg = be.get_awg()
#pg.send_waves_awg(awg, "sin", 1, 1000, 2000, 800, 1)
