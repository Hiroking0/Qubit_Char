import numpy as np
import matplotlib.pyplot as plt
import lmfit
from scipy import signal
from tkinter.filedialog import askopenfilename

# Define the file path
#add e^(j*phi) to
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

from matplotlib.ticker import ScalarFormatter
#file_path2 = '/content/drive/MyDrive/LPS/low_pow_read_at_-40dbm.txt'

fn = askopenfilename(filetypes=[("Text", "*.txt")])
file_path = fn

# Load the data from the text file, skipping the first row (header)
f,measured_s21 = np.genfromtxt(file_path, delimiter=',',dtype=np.complex_, skip_header=1)
#f1,measured_s211 = np.genfromtxt(file_path2, delimiter=',',dtype=np.complex_, skip_header=1)
#measured_s21= measured_s21*10**(46/20)
finit = f
measured_s21init = measured_s21
fig,ax = plt.subplots(figsize=(11,7))

#yhat = signal.savgol_filter(20*np.log10(np.abs(measured_s21)), window_length=800, polyorder=3, mode="nearest")



#plt.plot(f1, 20*np.log10(np.abs(measured_s211)))
ax.plot(f, 20*np.log10(np.abs(measured_s21)),label='Actual Data')
#ax.plot(f1,yhat,label='Smoothen Data')
ax.set_ylabel('|S21| (dBm)')
ax.set_xlabel('GHz')
ax.set_title('simulated measurement')
plt.legend()