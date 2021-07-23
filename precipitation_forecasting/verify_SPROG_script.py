import numpy as np
from batchcreator import load_fns_pysteps
from Evaluator import Evaluator
from pysteps_nowcast import sprog_forecast, anvil_forecast

val = np.load('datasets/test2020_3y_30m.npy', allow_pickle = True)

evaluator = Evaluator(save_after_n_samples = 1, nowcast_method = 'SPROG_test_bigdataset')

for val_row in val:
    R, R_target, metadata = load_fns_pysteps(val_row)
    R_f = sprog_forecast(R, metadata)

    leadtimes = [30, 60, 90]
    for i, leadtime in enumerate(leadtimes):
        evaluator.verify(R_target[i], R_f[i], leadtime=leadtime)