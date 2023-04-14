# qubit_measurement
This library has code for the following measurements/calibrations
- Amplitude sweep for qubit pulse
- Readout only pulse
- no-pulse/pulse measurement
- T1
- Rabi
- Ramsey

## Use of library
To use the library, modify the "general_config.yaml" file in the measurements folder to the desired parameters.
After setting the parameters, you must send the waves to the AWG. To do this run the send_XYZ.py file in either time_domain or calibration folders.
Running one of these files should look at the appropraite parameters in the config file and ignore the others.
After sending the waves, you can run one of four files in the measurements folder.

The "run_sequence.py" file will run the sequence according to the config file a single time.

The "single_rf_sweep.py" file will run a sweep based on the p1 parameter provided in the config file

The "double_rf_sweep.py" file will run a double sweep based on both parameters in the config

### Sweep parameters
The sweep parameters are combinations of two letters. The first letter either "p" for power or "w" for frequency. The second letter is either "q" for the RF generator corresponding to the qubit or "r" for the readout RF.

There is also "att" for the variable attenuator.

These parameters are inclusive of the initial value but NON-inclusive of the ending value.

The unit for frequency is GHz, for power it is DBm, for attenuation it is DB



