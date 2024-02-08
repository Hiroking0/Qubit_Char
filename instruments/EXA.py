import os
import sys
import matplotlib.pyplot as plt
import time
import numpy as np
sys.path.append("../../")
from lib import send_funcs as sf
from lib import run_funcs
from instruments.TekAwg import tek_awg as tawg
from lib import wave_construction as be
from qcodes.dataset import (
    Measurement,
    initialise_or_create_database_at,
    load_or_create_experiment,
)
from qcodes.instrument_drivers.Keysight import KeysightN9030B
from measurements.time_domain.run_sequence import eval_yaml
import yaml

f = open('./sbb_config.yaml','r')
params = yaml.safe_load(f)
params = eval_yaml(params)
f.close()

def get_inst():

    # Initialize Keysight N9030B
    driver = KeysightN9030B("n9010b", "TCPIP0::172.20.1.20::5025::SOCKET")
    awg = be.get_awg()
    driver.write(':INIT:CONT 1')
    # Set up the signal analyzer sweep
    sa = driver.sa
    
    return sa,awg


def getdata(sa,awg,start,stop,points):

    sa.setup_swept_sa_sweep(start,stop,points)

    # Initialize QCoDeS database
    tutorial_db_path = os.path.join(os.getcwd(), 'tutorial.db')
    initialise_or_create_database_at(tutorial_db_path)
    load_or_create_experiment(experiment_name='tutorial_exp', sample_name="no sample")

    # Define the measurement
    meas1 = Measurement()
    meas1.register_parameter(sa.trace)

    # Run the measurement and save results to the dataset
    with meas1.run() as datasaver:
        datasaver.add_result((sa.trace, sa.trace.get()))

    # Retrieve the dataset
    dataset = datasaver.dataset

    # Extract the raw data as a NumPy array
    raw_pow_data = dataset.get_parameter_data()['n9010b_sa_trace']['n9010b_sa_trace']
    raw_freq_data = dataset.get_parameter_data()['n9010b_sa_trace']['n9010b_sa_freq_axis']

    return (raw_freq_data,raw_pow_data)

def sweep(sa,awg,param,start,stop,points,sweep_start,sweep_end,sweep_step,text_freq = None, text_phase = None):

    freq = np.linspace(start,stop,points)
    sweep_param = np.arange(sweep_start,sweep_end + sweep_step,sweep_step)
    sa.setup_swept_sa_sweep(start,stop,points)

    # Initialize QCoDeS database
    tutorial_db_path = os.path.join(os.getcwd(), 'tutorial.db')
    initialise_or_create_database_at(tutorial_db_path)
    load_or_create_experiment(experiment_name='tutorial_exp', sample_name="no sample")

    # Define the measurement
    meas1 = Measurement()
    meas1.register_parameter(sa.trace)
    pow_mesh = np.empty((0,points))
    # Run the measurement and save results to the dataset
    with meas1.run() as datasaver:
        for y in sweep_param:
            print(y)
            if param == 'ch1':
                awg.set_offset(y,1)
            elif param == 'ch2':
                awg.set_offset(y,2)
            elif param == 'phase':
                name = "single_pulse"
                decimation = 1
                pattern_repeat = 1
                zero_length = 1000
                zero_multiple = 0
                readout_trigger_offset = 0
                #num_channels = 2
                num_patterns = awg.get_seq_length()
                if params['name'] == 'auto':
                    name = name + "_" + str(num_patterns)
                else:
                    name = name

                params[params['measurement']]['ssb_phase'] = str(y)
                params[params['measurement']]['ssb_freq'] = text_freq

                print(params[params['measurement']]['ssb_freq'])

                pg = sf.get_pg(params)

                pg.send_waves_awg(awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset, decimation)
                
                num_patterns = awg.get_seq_length()
                run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
                awg.run()
            
            elif param == 'freq':
                name = "single_pulse"
                decimation = 1
                pattern_repeat = 1
                zero_length = 1000
                zero_multiple = 0
                readout_trigger_offset = 0
                #num_channels = 2
                #print("--------sending--------")
                num_patterns = awg.get_seq_length()
                if params['name'] == 'auto':
                    name = name + "_" + str(num_patterns)
                else:
                    name = name

                params[params['measurement']]['ssb_freq'] = str(y)
                params[params['measurement']]['ssb_phase'] = text_phase

                #print('phase',params)

                pg = sf.get_pg(params)

                pg.send_waves_awg(awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset, decimation)
                
                num_patterns = awg.get_seq_length()
                run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
                #print("--------complete--------")
                awg.run()


            time.sleep(0.5)
            print(params[params['measurement']])
            datasaver.add_result((sa.trace, sa.trace.get()))

            # Retrieve the dataset
    dataset = datasaver.dataset

    # Extract the raw data as a NumPy array
    raw_pow_data = dataset.get_parameter_data()['n9010b_sa_trace']['n9010b_sa_trace']

    return freq*1e-9,sweep_param,raw_pow_data


