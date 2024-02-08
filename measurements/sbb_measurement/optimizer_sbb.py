import pyvisa
import sys
import yaml
sys.path.append("../../")
from instruments import EXA as exa
from lib import wave_construction as be

from measurements.time_domain.run_sequence import eval_yaml
from instruments.TekAwg import tek_awg as tawg
from lib import wave_construction as be
from lib import send_funcs as sf
from lib import run_funcs
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,Button,TextBox
from multiprocessing import Process, Manager
print("++++++++++++++++++++++++")

sa,awg = exa.get_inst()

# Connect to the instrument
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
ax_current_plot = plt.axes([0.2,0.28,0.7,0.7])
ax_sweep_options = plt.axes([0.75,0.1,0.1,0.03])

offset_ch1 = Slider(ax_slide_ch1,"offset_ch1 [V]",valmin= -2, valmax = 2.2, valinit= awg.get_offset(1), valstep= 0.001)
offset_ch2 = Slider(ax_slide_ch2,"offset_ch2 [V]",valmin= -2, valmax = 2.2, valinit= awg.get_offset(2), valstep= 0.001)
phase = TextBox(ax_slide_phase,"Phase_Shift [Deg]", initial= str(init_phase))
freq = TextBox(ax_slide_sbb_freq,"sbb_freq [GHz]", initial= str(init_freq))
sweep_options = Button(ax_sweep_options,"Sweep Options", hovercolor = 'green')

raw_freq_data,raw_pow_data = exa.getdata(sa,awg,2.9995e9,3.0005e9,20000)
scatter_plot = ax_current_plot.scatter(raw_freq_data, raw_pow_data)
ax_current_plot.set_xlabel('Frequency (Hz)')
ax_current_plot.set_ylabel('Power (dBm)')
ax_current_plot.set_title('Power vs Frequency')

def sweep_button(val):
    sweep = Process(target = opt.start())

    return


def update_plot():
    raw_freq_data,raw_pow_data = exa.getdata(sa,awg,2.9995e9,3.0005e9,20000)
    ax_current_plot.clear()  # clearing the axes
    ax_current_plot.scatter(raw_freq_data, raw_pow_data)  # creating new scatter chart with updated data
    ax_current_plot.figure.canvas.draw_idle()

def update_ch1_offset(val):
    awg.set_offset(offset_ch1.val,1)
    update_plot()

def update_ch2_offset(val):
    awg.set_offset(offset_ch2.val,2)
    update_plot()

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
    update_plot()

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
    update_plot()

offset_ch1.on_changed(update_ch1_offset)
offset_ch2.on_changed(update_ch2_offset)
phase.on_submit(update_phase)
freq.on_submit(update_freq)
#sweep_options.on_clicked(sweep_button)
plt.show()