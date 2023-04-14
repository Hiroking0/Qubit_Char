
import sys
sys.path.append("../")

#import npp_funcs as nppf
from lib import wave_construction as be
from lib import send_funcs as sf
import yaml
from lib import run_funcs


if __name__ == "__main__":
    f = open('./general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    name = params['name']
    pattern_repeat = params['pattern_repeat']
    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    readout_trigger_offset = params['readout_trigger_offset']

    pg = sf.get_pg(params)

    awg = be.get_awg()
    
    pg.send_waves_awg(awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset)
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)

    awg.close()
