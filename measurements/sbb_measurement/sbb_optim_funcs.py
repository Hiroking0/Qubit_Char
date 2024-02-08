import pyvisa
import sys
import yaml
import numpy as np
sys.path.append("../../")
from instruments import EXA as exa
from lib import wave_construction as be
from measurements.time_domain.run_sequence import eval_yaml
from instruments.TekAwg import tek_awg as tawg
from lib import wave_construction as be
from lib import send_funcs as sf
from lib import run_funcs
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager
from matplotlib.widgets import Slider,Button,TextBox,CheckButtons

# Connect to the instrument
awg = be.get_awg()
f = open('./sbb_config.yaml','r')
params = yaml.safe_load(f)
params = eval_yaml(params)
f.close()
init_phase = params[params['measurement']]['ssb_phase']
init_freq = params[params['measurement']]['ssb_freq']

fig_controls = plt.figure(figsize=(6,10))
ax_slide_ch1_start = plt.axes([0.3,0.8,0.1,0.03])
ax_slide_ch1_end = plt.axes([0.55,0.8,0.1,0.03])
ax_slide_ch1_step = plt.axes([0.85,0.8,0.1,0.03])

ax_slide_ch2_start = plt.axes([0.3,0.85,0.1,0.03])
ax_slide_ch2_end = plt.axes([0.55,0.85,0.1,0.03])
ax_slide_ch2_step = plt.axes([0.85,0.85,0.1,0.03])

ax_slide_phase_start = plt.axes([0.3,0.9,0.1,0.04])
ax_slide_phase_end = plt.axes([0.55,0.9,0.1,0.04])
ax_slide_phase_step= plt.axes([0.85,0.9,0.1,0.04])

ax_slide_sbb_freq_start = plt.axes([0.3,0.95,0.1,0.04])
ax_slide_sbb_freq_end = plt.axes([0.55,0.95,0.1,0.04])
ax_slide_sbb_freq_step = plt.axes([0.85,0.95,0.1,0.04])

ax_sweep_start = plt.axes([0.75,0.05,0.19,0.03])
ax_sweep_param = plt.axes([0.75,0.1,0.19,0.13])

ax_slide_ch1 = plt.axes([0.2,0.55,0.5,0.03])
ax_slide_ch2 = plt.axes([0.2,0.6,0.5,0.03])
ax_slide_phase = plt.axes([0.2,0.65,0.1,0.04])
ax_slide_sbb_freq = plt.axes([0.2,0.7,0.1,0.04])

fig = plt.figure(figsize=(11,6))
ax_current_plot = plt.axes([0.1,0.28,0.7,0.7])
ax_current_cmap = plt.axes([0.85,0.28,0.05,0.7])
#ax_save = plt.axes([0.65,0.18,0.1,0.04])

ch1_start = TextBox(ax_slide_ch1_start,"Start offset ch1 [V]", initial= '-1')
ch1_end = TextBox(ax_slide_ch1_end,"End [V]", initial= '1')
ch1_step = TextBox(ax_slide_ch1_step,"Step [V]", initial= '0.5')

ch2_start = TextBox(ax_slide_ch2_start,"Start offset ch2 [V]", initial= '')
ch2_end = TextBox(ax_slide_ch2_end,"End [V]", initial= '')
ch2_step = TextBox(ax_slide_ch2_step,"Step [V]", initial= '')

phase_start = TextBox(ax_slide_phase_start,"Start Phase Shift [Deg]", initial= '85')
phase_end = TextBox(ax_slide_phase_end,"End [Deg]", initial= '95')
phase_step = TextBox(ax_slide_phase_step,"Step [Deg]", initial= '1')

freq_start = TextBox(ax_slide_sbb_freq_start,"Start sbb freq [GHz]", initial= '0')
freq_end = TextBox(ax_slide_sbb_freq_end,"End [GHz]", initial= '0.06')
freq_step = TextBox(ax_slide_sbb_freq_step,"Step [GHz]", initial= '0.003')

sweep_start = Button(ax_sweep_start,"Initiate Sweep", hovercolor = 'green')
sweep_params = CheckButtons(ax_sweep_param, ('ch1 offset','ch2 offset','phase','ssb freq'), (False, False, False,False))

offset_ch1 = Slider(ax_slide_ch1,"offset_ch1 [V]",valmin= -2, valmax = 2.2, valinit= awg.get_offset(1), valstep= 0.001)
offset_ch2 = Slider(ax_slide_ch2,"offset_ch2 [V]",valmin= -2, valmax = 2.2, valinit= awg.get_offset(2), valstep= 0.001)
phase = TextBox(ax_slide_phase,"Phase_Shift [Deg]", initial= str(init_phase))
freq = TextBox(ax_slide_sbb_freq,"sbb_freq [GHz]", initial= str(init_freq))

ax_current_plot.set_xlabel('Freqeuncy (GHz)')
ax_current_plot.set_ylabel('Selected Param')
ax_current_cmap.set_ylabel('Power (dBm)')

sa,awg = exa.get_inst()

plot_freq_start = 2.7e9
plot_freq_end = 3.3e9

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

    
    #print('phase',params)

    pg = sf.get_pg(params)

    pg.send_waves_awg(awg, name, pattern_repeat, zero_length, zero_multiple, readout_trigger_offset, decimation)
    
    num_patterns = awg.get_seq_length()
    run_funcs.initialize_awg(awg, num_patterns, pattern_repeat, decimation)
    print("--------complete--------")
    awg.run()


def labeling(label):
    if label == 'ch1':
        param = 'ch1'
        ax_current_plot.set_ylabel("ch1 (V)")
    elif label == 'ch2':
        param = 'ch2'
        ax_current_plot.set_ylabel("ch2 (V)")
    elif label == 'phase':
        param = 'phase'
        ax_current_plot.set_ylabel("Phase (deg)")
    elif label == 'freq':
        param = 'freq'
        ax_current_plot.set_ylabel("Frequency Offset (MHz)")

    

def start_sweeping(val):

    status = sweep_params.get_status()
    if status == [1,0,0,0]:
        param = 'ch1'
        start = ch1_start
        end = ch1_end
        step = ch1_step
    elif status == [0,1,0,0]:
        param = 'ch2'
        start = ch2_start
        end = ch2_end
        step = ch2_step
    elif status == [0,0,1,0]:
        param = 'phase'
        start = phase_start
        end = phase_end
        step = phase_step
    elif status == [0,0,0,1]:
        param = 'freq'
        start = freq_start
        end = freq_end
        step = freq_step
    raw_freq_data = []
    sweep_param = []
    pow_mesh = []
    raw_freq_data,sweep_param,pow_mesh = exa.sweep(sa,awg,param,plot_freq_start,plot_freq_end,2000,
                                            float(start.text),float(end.text),float(step.text),float(freq.text),float(phase.text))
    if param == 'freq':
        sweep_param = sweep_param*1000
    x,y = np.meshgrid(raw_freq_data,sweep_param)
    CS = ax_current_plot.contourf(x, y, pow_mesh, levels=100)

    # Add a colorbar to ax_current_cmap
    cbar = plt.colorbar(CS, cax=ax_current_cmap)
    cbar.set_label('Power (dBm)', rotation=270)

    ax_current_plot.set_xlim(plot_freq_start*1e-9,plot_freq_end*1e-9)
    if param == 'freq':
        ax_current_plot.set_ylim(float(start.text)*1000,float(end.text)*1000)
    else:
        ax_current_plot.set_ylim(float(start.text),float(end.text))
    
    # Optionally, add colorbar label
    fig.canvas.draw()


sweep_params.on_clicked(labeling)
sweep_start.on_clicked(start_sweeping)
offset_ch1.on_changed(update_ch1_offset)
offset_ch2.on_changed(update_ch2_offset)
phase.on_submit(update_phase)
freq.on_submit(update_freq)
plt.show()
