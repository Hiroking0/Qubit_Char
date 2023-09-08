
import sys
sys.path.append("../")

#import npp_funcs as nppf
from lib import wave_construction as be
from lib import send_funcs as sf
import yaml
from lib import run_funcs
def int_eval(data):
    return eval(str(data))
if __name__ == "__main__":
    f = open('./sbb_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    name = params['name']
    decimation = int_eval(params['decimation'])
    pattern_repeat = int_eval(params['pattern_repeat'])
    zero_length = int_eval(params['zero_length'])
    zero_multiple = int_eval(params['zero_multiple'])
    readout_trigger_offset = int_eval(params['readout_trigger_offset'])
    #num_channels = 2
    awg = be.get_awg()

    num_patterns = awg.get_seq_length()
    if params['name'] == 'auto':
        name = params['measurement'] + "_" + str(num_patterns)
    else:
        name = params['name']

    pg = sf.get_pg(params)

    awg = be.get_awg()
    pg.send_waves_awg(awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset, decimation)
    
    num_patterns = awg.get_seq_length()
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)

    awg.close()
