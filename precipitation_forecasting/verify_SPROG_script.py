import numpy as np
from batchcreator import load_fns_pysteps
from validation import Evaluator
from pysteps_nowcast import sprog_forecast, get_mask_rtcor
import config
import os

val = np.load('datasets/test2020_3y_30m.npy', allow_pickle = True)
nowcast_method = 'SPROG_test' 
evaluator = Evaluator(save_after_n_samples = 1, nowcast_method = nowcast_method)

mask = get_mask_rtcor()

# Create directory to save predictions
pred_path = config.dir_pred + nowcast_method + '/' 
if not os.path.exists(pred_path):
    os.makedirs(pred_path) 
    
for val_row in val:
    R, R_target, metadata, metadata_target = load_fns_pysteps(val_row)
    R_forecast = sprog_forecast(R, metadata, mask)

    leadtimes = [30, 60, 90]
    timestamps = metadata_target['timestamps']
    
    for i, leadtime in enumerate(leadtimes):
        timestamp = timestamps[i].strftime("%Y%m%d%H%M")
        # save forecast:
        fn_forecast = timestamp+'_lt{}'.format(leadtime)
        np.save(pred_path+fn_forecast, R_forecast[i])
                
        evaluator.verify(R_target[i], R_forecast[i], leadtime=leadtime)