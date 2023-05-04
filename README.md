# qubit_measurement
This library has code for the following measurements/calibrations
- Readout only pulse
- no-pulse/pulse measurement
- T1
- Rabi
- Ramsey
- Echo

### Terminology 
pattern: one entry (of all four channels) on the AWG. For the no-pulse pulse measurement, one pattern would be an entry including just
the readout pulse, one pattern would be the entry including the pulse and readout.

sequence: The group of patterns that make up a measurement. That would be all n patterns making up a rabi, ramsey, etc...

## Use of library
To use the library, modify the "general_config.yaml" file in the measurements folder to the desired parameters.
After setting the parameters, you must send the waves to the AWG. To do this run the send_pulse.py file. This file will send the sequence corresponding
to the "measurement" parameter in the config file. The possible parameters are listed at the top of the file.


Running one of these files should look at the appropraite parameters in the config file and ignore the others.
After sending the waves, you can run one of four files in the measurements folder.

The "run_sequence.py" file will run the sequence according to the config file a single time.

The "single_rf_sweep.py" file will run a sweep based on the p1 parameter provided in the config file

The "double_rf_sweep.py" file will run a double sweep based on both parameters in the config

### Sweep parameters
The sweep parameters are combinations of two letters. The first letter either "p" for power or "w" for frequency. The second letter is either "q" for the RF generator corresponding to the qubit or "r" for the readout RF.

There are also "r_att" and "q_att" for the readout and qubuit variable attenuators respectively.

These parameters are inclusive of the initial value but NON-inclusive of the ending value.

The unit for frequency is GHz, for power it is DBm, for attenuation it is DB

## lib files
### data_process
This file is what handles all the post-acquisition data processing.
There are two functions that can be run after an acquisition. If the total amount of data is small enough (<1000 triggers)
the raw data is saved for each acquisition in the data.bin file.
the function plot_all gets called which plots the average of each readout pulse for each pattern, along with the IQ data and histograms.
If the data is larger than 1000 triggers, only the average of each trigger is stored (the average voltage value from average_start to average_start + average_length)
The function plot_np_file is called. This function plots the pattern number vs the average voltage for chA, B, mags, etc...

### wave_construction
This library is for creating and sending patterns and sequences to the AWG.
The function get_awg should be used to get the AWG instance, init_awg is used to set the correct parameters for each pattern entry on the AWG.
The classes for pulses represent a square wave, and the parameters are given in the constructors.
The key function for each of the pulses is the make() function. This function takes the parameters and creates a 2D array from them.
The dimensions of the array are \[pattern #\]\[sample #\]. For a single pattern, the first dimension length is 1.
For a swept pulse, the first dimesion is the number of patterns.

The readout pulse class has a few assumptions:
it will always be the last pulse in the pattern,
it will not be swept,
it will determine the length of the pattern (excluding the wait time after)

When creating a sequence to send to the AWG, create all the pulses you want then put them in a list (in no particular order).
Create a PulseGroup with that list as a parameter and call send_waves_awg().

There are also show() functions for Pulses and PulseGroups to display what will be sent.

The deadtime after a readout pulse is determined by the zero_length and zero_multiple parameters.
It will add a portion of 0s with the length zero_length at the end of the readout pulse, and repeat that portion of zeros by zero_multiple times.
For example if you use zero_length = 1000 and zero_multiple = 2000. The total dead time will be 1000*2000 samples. (for 1GS/s this will be 2ms)

### Adding a new pattern
Use the sweep_pulse class as a template. You must have a constructor with all the necessary parameters for the pulse.
The make function should return a 2d array for each of the channels as described above




## Fitting
These files are used to plot saved data with a fit line overlayed with it. 
The only difference between these files are the parameters it looks at to determine the x axis,
and the function it uses to fit.


### disp_data
This file has a few functions.
disp_sequence is used for displaying the npy files that are saved after individual run_sequence runs
disp_single_sweep is used for reading the csv files output by single_rf_sweep.py runs.
It shows pattern_number vs sweep_parameter.

show_sweep_output shows a single sweep csv file but the axes are sweep_parameter vs voltage. All the patterns are overlayed on top of each other.
This would be best for npp measurement, the more patterns the more cluttered it gets.


disp_3_chevrons is used specifically for showing only 3 slices of the chevron measurement.
The three slices can be chosen in the parameters.




