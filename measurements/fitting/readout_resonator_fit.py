import matplotlib.pyplot as plt
import numpy as np
import sys
import lmfit
import tkinter.filedialog as filedialog
from matplotlib.widgets import Slider,Button,TextBox, CheckButtons
import time
sys.path.append("../../")
from instruments import qfreq_qpow_sweep as read
def linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag):
    Q_e = Q_e_real + 1j*Q_e_imag
    return 1 - (Q * Q_e**-1 / (1 + 2j * Q * (f - f_0) / f_0))

class ResonatorModel(lmfit.model.Model):
    __doc__ = "resonator model" + lmfit.models.COMMON_INIT_DOC

    def __init__(self, *args, **kwargs):
        # pass in the defining equation so the user doesn't have to later
        super().__init__(linear_resonator, *args, **kwargs)

        self.set_param_hint('Q', min=0)  # enforce Q is positive

    def guess(self, data, f=None, **kwargs):
        verbose = kwargs.pop('verbose', None)
        if f is None:
            return
        argmin_s21 = np.abs(data).argmin()
        fmin = f.min()
        fmax = f.max()
        f_0_guess = f[argmin_s21]  # guess that the resonance is the lowest point
        Q_min = 0.1 * (f_0_guess/(fmax-fmin))  # assume the user isn't trying to fit just a small part of a resonance curve
        delta_f = np.diff(f)  # assume f is sorted
        min_delta_f = delta_f[delta_f > 0].min()
        Q_max = f_0_guess/min_delta_f  # assume data actually samples the resonance reasonably
        Q_guess = np.sqrt(Q_min*Q_max)  # geometric mean, why not?
        Q_e_real_guess = Q_guess/(1-np.abs(data[argmin_s21]))
        if verbose:
            print(f"fmin={fmin}, fmax={fmax}, f_0_guess={f_0_guess}")
            print(f"Qmin={Q_min}, Q_max={Q_max}, Q_guess={Q_guess}, Q_e_real_guess={Q_e_real_guess}")
        params = self.make_params(Q=Q_guess, Q_e_real=Q_e_real_guess, Q_e_imag=0, f_0=f_0_guess)
        params[f'{self.prefix}Q'].set(min=Q_min, max=Q_max)
        params[f'{self.prefix}f_0'].set(min=fmin, max=fmax)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)
    
resonator = ResonatorModel()
true_params = resonator.make_params(f_0=100, Q=10000, Q_e_real=9000, Q_e_imag=-9000)

###############################################################params
center = 7.09161e9 #in Hz
span = 500e3 #in Hz
points=2001
pow = -69 #in dbm
start = center - span/2
stop=center + span/2
f = np.linspace(start*1e-9, stop*1e-9, points)
measured_s21 = read.getdata(start,stop,points,pow)
fig,ax = plt.subplots(1,1,figsize=(12,8))
line, = ax.plot(f, 20*np.log10(np.abs(measured_s21)),label = 'meas data')
ax.set_ylabel('|S21| (dB)')
ax.set_xlabel('GHz')
ax.set_title('simulated measurement')
'''plt.figure()
plt.scatter(np.real(measured_s21),np.imag(measured_s21))'''
ax_center = plt.axes([0.1,0.01,0.15, 0.03])
ax_span = plt.axes([0.33, 0.01, 0.1, 0.03])
axpoints = plt.axes([0.48, 0.01, 0.1, 0.03])
ax_pow = plt.axes([0.66, 0.01, 0.1, 0.03])
ax_update = plt.axes([0.8, 0.01, 0.05, 0.03])
ax_save = plt.axes([0.9, 0.01, 0.05, 0.03])
text_center = str(center*1e-9)
text_span = str(span*1e-3)
center = TextBox(ax_center,'Center(GHz)', initial='7.0859166')
span = TextBox(ax_span,'Span(MHz)', initial='1')
point = TextBox(axpoints,'Points', initial='2001')
pow = TextBox(ax_pow,'Power(dBm)', initial='-10')
update = CheckButtons(ax_update,['Update'])
saves = Button(ax_save,"Save",hovercolor = 'green')

def save(event):
    centers = eval(str(center.text))*1e9 #in Hz
    spans = eval(str(span.text))*1e6 #in Hz
    points=eval(str(point.text))
    pows = eval(str(pow.text))
    
    start = centers - spans/2
    stop=centers + spans/2
    
    f = np.linspace(start, stop, points)
    measured_s21 = read.getdata(start,stop,points,pows)
    line.set_data(f, 20*np.log10(np.abs(measured_s21)))
    ax.set_xlim([start-0.05*spans ,stop+ 0.05*spans])
    yplot = 20*np.log10(np.abs(measured_s21))
    diff = abs(min(yplot) - max(yplot))
    ax.set_ylim([min(yplot) - 0.05*diff ,max(yplot + 0.05*diff)])
    fig.canvas.draw_idle()

    file_path = filedialog.askdirectory()
    name = 'res_plot.txt'
    ff = file_path + '/' + name
    # Save the data to a text file
    data = [f,measured_s21]
    header = 'Column1,Column2'
    np.savetxt(ff, data, delimiter=',', header=header)

def update_fit(event):
    print("updated")
    centers = eval(str(center.text))*1e9 #in Hz
    spans = eval(str(span.text))*1e6 #in Hz
    points=eval(str(point.text))
    pows = eval(str(pow.text))
    
    start = centers - spans/2
    stop=centers + spans/2
    
    f = np.linspace(start, stop, points)
    measured_s21 = read.getdata(start,stop,points,pows)
    line.set_data(f, 20*np.log10(np.abs(measured_s21)))
    ax.set_xlim([start-0.05*spans ,stop+ 0.05*spans])
    yplot = 20*np.log10(np.abs(measured_s21))
    diff = abs(min(yplot) - max(yplot))
    ax.set_ylim([min(yplot) - 0.05*diff ,max(yplot + 0.05*diff)])
    fig.canvas.draw_idle()
    plt.show()
    print('end')
    return

def on_checkbox_change(label):
    global running
    running = not running
    if running:
        run_function()

# Function to run the update_function at a 2-second interval
def run_function():
    while running:
        update_fit()
        time.sleep(2)
        print(running)

# Initialize the checkbox state
running = False


update.on_clicked(update_fit)
saves.on_clicked(save)

plt.show()
#__________________________________________________________
print(measured_s21.dtype)
guess = resonator.guess(measured_s21, f=f*1e-6, verbose=True)
print("_________________")
print(guess)
result = resonator.fit(measured_s21, params=guess, f=f*1e-6, verbose=True)

print(result.fit_report() + '\n')
result.params.pretty_print()

#____________________________________________________________________________________
def plot_ri(data, *args, **kwargs):
    plt.plot(data.real, data.imag, *args, **kwargs)


fit_s21 = resonator.eval(params=result.params, f=f*1e-6)
guess_s21 = resonator.eval(params=guess, f=f*1e-6)

plt.figure()
plot_ri(measured_s21, '.')
plot_ri(fit_s21, '.-', label='best fit')
plot_ri(guess_s21, '--', label='initial fit')
plt.legend()
plt.xlabel('Re(S21)')
plt.ylabel('Im(S21)')

plt.figure()
plt.plot(f*1e-9, 20*np.log10(np.abs(measured_s21)), '.')
plt.plot(f*1e-9, 20*np.log10(np.abs(fit_s21)), '.-', label='best fit')
plt.plot(f*1e-9, 20*np.log10(np.abs(guess_s21)), '--', label='initial fit')
plt.legend()
plt.ylabel('|S21| (dB)')
plt.xlabel('GHz')
plt.show()
"""
kappa/2Pi = F0/Q 

"""

