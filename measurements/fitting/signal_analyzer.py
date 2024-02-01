import os

from qcodes.dataset import (
    Measurement,
    initialise_or_create_database_at,
    load_or_create_experiment,
    plot_by_id,
    plot_dataset,
)
from qcodes.instrument_drivers.Keysight import KeysightN9030B

driver = KeysightN9030B("n9010b","TCPIP0::172.20.1.20::5025::SOCKET")
driver.IDN()
sa = driver.sa
sa.setup_swept_sa_sweep(start=200, stop= 10e3, npts=2001)
tutorial_db_path = os.path.join(os.getcwd(), 'tutorial.db')
initialise_or_create_database_at(tutorial_db_path)
load_or_create_experiment(experiment_name='tutorial_exp', sample_name="no sample")
meas1 = Measurement()
meas1.register_parameter(sa.trace)
with meas1.run() as datasaver:
    datasaver.add_result((sa.trace, sa.trace.get()))

dataset = datasaver.dataset
plot_dataset(dataset)