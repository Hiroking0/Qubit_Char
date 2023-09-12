from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
from qcodes import VisaInstrument
import numpy as np
import sys
sys.path.append("../")
from lib import send_funcs as sf
from measurements.run_sequence import eval_yaml
from lib import wave_construction as be
#import npp_funcs as nppf
import yaml
def int_eval(data):
    return eval(str(data))
if __name__ == "__main__":
    f = open('./sbb_config.yaml','r')
    params = yaml.safe_load(f)
    params = eval_yaml(params)
    f.close()
    decimation = int_eval(params['decimation'])

    pg = sf.get_pg(params)

    no_subplot_measurements = ["echo", "echo_1ax"]
    subplots = not params['measurement'] in no_subplot_measurements

    #subplots = True
    #if params['measurement'] == "echo" or params['measurement'] == "echo_1ax":
    #    subplots = False
    if params['Pulse_without_ssb_phase']==params['Pulse_with_ssb_phase']:
        raise Exception("Logic error in channel selection")
    #pg.show(decimation, subplots)
    (c1, c1m1, c2, c2m1, c2m2, c3, c4) = pg.make()
    waveforms = [c1,c2,c3,c4]
    m2s = [c2m1, c2m2]
    m1s = [c1m1, c1m1]
    pattern_repeat = params['pattern_repeat']
    seq_repeat = params['seq_repeat']
    Tektronix_AWG5014.make_and_save_awg_file()
    Tektronix_AWG5014.make_and_save_awg_file(Tektronix_AWG5014,
                                    waveforms=waveforms,
                                    m1s=m1s,
                                    m2s=m2s,
                                    nreps=seq_repeat,
                                    trig_waits=0,
                                    goto_states=1,
                                    jump_tos=0,
                                    channels=[1,2,3,4],
                                    preservechannelsettings=True)
