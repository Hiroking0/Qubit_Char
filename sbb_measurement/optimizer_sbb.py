import pyvisa
import sys
import yaml
sys.path.append("../")
from lib import wave_construction as be
from measurements.run_sequence import eval_yaml
from instruments.TekAwg import tek_awg as tawg
from lib import wave_construction as be
from lib import send_funcs as sf
from lib import run_funcs
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,Button,TextBox
# Connect to the instrument

'''import pyvisa
rm = pyvisa.ResourceManager()
inst = rm.open_resource('TCPIP0::::5025::SOCKET')
inst.query('*IDN?')


inst.read_termination = '\n'
inst.query_delay = 1e-3
inst.timeout = 10000

inst.write(':FREQ:STAR: 200 MHz')
inst.write(':FREQ:STOP: 800 MHz') '''
awg = be.get_awg()
f = open('./sbb_config.yaml','r')
params = yaml.safe_load(f)
params = eval_yaml(params)
f.close()
init_phase = params[params['measurement']]['ssb_phase']
init_freq = params[params['measurement']]['ssb_freq']
ax_slide_ch1 = plt.axes([0.2,0.1,0.5,0.03])
ax_slide_ch2 = plt.axes([0.2,0.05,0.5,0.03])
ax_slide_phase = plt.axes([0.2,0.15,0.1,0.04])
ax_slide_sbb_freq = plt.axes([0.2,0.2,0.1,0.04])
offset_ch1 = Slider(ax_slide_ch1,"offset_ch1 [V]",valmin= -2, valmax = 2, valinit= awg.get_offset(1), valstep= 0.001)
offset_ch2 = Slider(ax_slide_ch2,"offset_ch2 [V]",valmin= -2, valmax = 2, valinit= awg.get_offset(2), valstep= 0.001)
phase = TextBox(ax_slide_phase,"Phase_Shift [Deg]", initial= str(init_phase))
freq = TextBox(ax_slide_sbb_freq,"sbb_freq [GHz]", initial= str(init_freq))
def update_ch1_offset(val):
    awg.set_offset(offset_ch1.val,1)
    
def update_ch2_offset(val):
    awg.set_offset(offset_ch2.val,2)

def update_phase(val):
    name = params['name']
    decimation = params['decimation']
    pattern_repeat = params['pattern_repeat']
    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    readout_trigger_offset = params['readout_trigger_offset']
    #num_channels = 2
    print("--------sending--------")
    num_patterns = awg.get_seq_length()
    if params['name'] == 'auto':
        name = params['measurement'] + "_" + str(num_patterns)
    else:
        name = params['name']
    
    params[params['measurement']]['ssb_phase'] = str(phase.text)
    print(params[params['measurement']])
    print('meas',params[params['measurement']]['ssb_phase'])

    #print('phase',params)

    pg = sf.get_pg(params)

    pg.send_waves_awg(awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset, decimation)
    
    num_patterns = awg.get_seq_length()
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    print("--------complete--------")
    awg.run()

def update_freq(val):
    name = params['name']
    decimation = params['decimation']
    pattern_repeat = params['pattern_repeat']
    zero_length = params['zero_length']
    zero_multiple = params['zero_multiple']
    readout_trigger_offset = params['readout_trigger_offset']
    #num_channels = 2
    print("--------sending--------")
    num_patterns = awg.get_seq_length()
    if params['name'] == 'auto':
        name = params['measurement'] + "_" + str(num_patterns)
    else:
        name = params['name']

    params[params['measurement']]['ssb_freq'] = str(freq.text)
    print(params[params['measurement']])
    print('meas',params[params['measurement']]['ssb_freq'])

    
    #print('phase',params)

    pg = sf.get_pg(params)

    pg.send_waves_awg(awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset, decimation)
    
    num_patterns = awg.get_seq_length()
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    print("--------complete--------")
    awg.run()

offset_ch1.on_changed(update_ch1_offset)
offset_ch2.on_changed(update_ch2_offset)
phase.on_submit(update_phase)
freq.on_submit(update_freq)
plt.show()