import matplotlib.pyplot as plt
import numpy as np
import sys
import lmfit
import tkinter.filedialog as filedialog
from matplotlib.widgets import TextBox, CheckButtons, Button
from matplotlib.animation import FuncAnimation
sys.path.append("../../")
from instruments import qfreq_qpow_sweep as read

# Your initial parameters
center = 7.09161e9
span = 500e3
points = 2001
pow = -69
start = center - span / 2
stop = center + span / 2

# Initial data
measured_s21, f = read.justgetdata()

# Create the main figure for the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
line, = ax.plot(f, 20 * np.log10(np.abs(measured_s21)), label='meas data')
ax.set_ylabel('|S21| (dB)')
ax.set_xlabel('GHz')
ax.set_title('Simulated Measurement')

# Create a separate figure for the widgets with the specified arrangement
fig_controls = plt.figure(figsize=(4, 7))

ax_center = plt.axes([0.25, 0.95, 0.2, 0.03])
ax_span = plt.axes([0.25, 0.9, 0.15, 0.03])
axpoints = plt.axes([0.25, 0.85, 0.15, 0.03])
ax_pow = plt.axes([0.25, 0.8, 0.15, 0.03])
ax_bandwidth = plt.axes([0.35, 0.75, 0.15, 0.03])

center_textbox = TextBox(ax_center, 'Center(GHz)', initial='7.0859166')
span_textbox = TextBox(ax_span, 'Span(MHz)', initial='1')
point_textbox = TextBox(axpoints, 'Points', initial='2001')
pow_textbox = TextBox(ax_pow, 'Power(dBm)', initial='-10')
bandwidth_textbox = TextBox(ax_bandwidth, 'IF Bandwidth(kHz)', initial='100')

ax_update = plt.axes([0.6, 0.01, 0.2, 0.03])
ax_save = plt.axes([0.8, 0.01, 0.1, 0.03])

update_button = CheckButtons(ax_update, ['Update'])
save_button = Button(ax_save, "Save", hovercolor='green')

# Global variable for animation control
running = False

def update_fit(frame):
    global running
    if running:
        centers = eval(center_textbox.text) * 1e9
        spans = eval(span_textbox.text) * 1e6
        points = eval(point_textbox.text)
        pows = eval(pow_textbox.text)
        bandwidths = eval(bandwidth_textbox.text)
        start = centers - spans / 2
        stop = centers + spans / 2

        f = np.linspace(start, stop, points)
        measured_s21 = read.getdata(start, stop, points, pows, bandwidths)
        line.set_data(f, 20 * np.log10(np.abs(measured_s21)))
        ax.set_xlim([start - 0.05 * spans, stop + 0.05 * spans])
        yplot = 20 * np.log10(np.abs(measured_s21))
        diff = abs(min(yplot) - max(yplot))
        ax.set_ylim([min(yplot) - 0.05 * diff, max(yplot + 0.05 * diff)])
        plt.pause(0.01)

def on_checkbox_change(label):
    global running
    running = not running

def save(event):
    centers = eval(center_textbox.text) * 1e9
    spans = eval(span_textbox.text) * 1e6
    points = eval(point_textbox.text)
    pows = eval(pow_textbox.text)
    bandwidths = eval(bandwidth_textbox.text)
    start = centers - spans / 2
    stop = centers + spans / 2

    f = np.linspace(start, stop, points)
    measured_s21 = read.getdata(start, stop, points, pows, bandwidths)
    line.set_data(f, 20 * np.log10(np.abs(measured_s21)))
    ax.set_xlim([start - 0.05 * spans, stop + 0.05 * spans])
    yplot = 20 * np.log10(np.abs(measured_s21))
    diff = abs(min(yplot) - max(yplot))
    ax.set_ylim([min(yplot) - 0.05 * diff, max(yplot + 0.05 * diff)])
    plt.pause(0.01)

    file_path = filedialog.askdirectory()
    name = 'res_plot.txt'
    ff = file_path + '/' + name
    # Save the data to a text file
    data = np.column_stack((f, measured_s21))
    header = 'Column1,Column2'
    np.savetxt(ff, data, delimiter=',', header=header)

# Set up animation
ani = FuncAnimation(fig, update_fit, blit=False)

# Connect callbacks
update_button.on_clicked(on_checkbox_change)
save_button.on_clicked(save)

plt.show()
