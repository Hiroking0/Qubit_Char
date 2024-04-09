import os
import sys
import matplotlib.pyplot as plt
import time
import numpy as np
from qcodes.dataset import (
    Measurement,
    initialise_or_create_database_at,
    load_or_create_experiment,
)
import yaml

sys.path.append("../")
from instruments.TekAwg import tek_awg as tawg

from qcodes.instrument_drivers.tektronix import TektronixAWG5014


# Create an instance of TektronixAWG5014 with custom resource properties
#inst = TektronixAWG5014("AWG", "172.20.1.5", timeout=10000, terminator='\n', query_delay=1e-3)


ip = '172.20.1.5'
port=5000

awg = tawg.TekAwg.connect_raw_visa_socket(ip, port)
print(awg.get_trig_interval())





