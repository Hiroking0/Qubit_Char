# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:44:38 2023

@author: lqc
"""

import tkinter as tk
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory
import numpy as np
import sys
sys.path.append("../../")
from lib import data_process as dp
import matplotlib.pyplot as plt
import os
import json
import pandas
import fit_rabi
import pickle as pkl
from fit_rabi import  fit_rabi
from scipy.optimize import curve_fit
from multiprocessing import Process, Manager
from qcodes.dataset import (
    Measurement,
    initialise_or_create_database_at,
    load_by_run_spec,
    load_from_netcdf,
    load_or_create_experiment,
    plot_dataset,)
file_path = askopenfilename(filetypes=[("NetCDF files", "*.nc")])
data = load_from_netcdf(file_path)
type(data)
plot_dataset(data)
plt.show()