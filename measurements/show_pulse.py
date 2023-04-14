
import sys
sys.path.append("../")
from lib import send_funcs as sf
#import npp_funcs as nppf
from lib import wave_construction as be
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
    decimation = params['decimation']

    pg = sf.get_pg(params)
    subplots = True
    if params['measurement'] == "echo" or params['measurement'] == "echo_1ax":
        subplots = False

    pg.show(decimation, subplots)
