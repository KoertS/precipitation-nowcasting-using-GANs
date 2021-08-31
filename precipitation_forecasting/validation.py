from pysteps.verification.detcatscores import det_cat_fct_init, det_cat_fct_accum, det_cat_fct_compute, det_cat_fct_merge
from pysteps.verification.detcontscores import det_cont_fct_init, det_cont_fct_accum, det_cont_fct_compute
from pysteps.verification.spatialscores import fss_init, fss_accum, fss_compute
import numpy as np
from batchcreator import DataGenerator, undo_prep
from model_builder import GAN
import tensorflow as tf
from tqdm import tqdm
from pysteps_nowcast import get_mask_rtcor
import config
import os

class Evaluator:
    '''
    This functions is used to validate a models predictions.
    Categorical scores and continues errors are accumulated in dictionaries.
    When validation is finished, the metrics can be computed by using these dictionaries.
    nowcast_method: indicates with what model the nowcast were made
    thresholds: precipitation threshold to use for the categorical scores
    leadtimes: the leadtimes of the predictions
    save_after_n_sample: if higher than 0, the evaluator will save its dictionary after it has seen n samples. 
    scales: The spatial scales used to compute FSS at different spatial levels
    '''
    def __init__(self, nowcast_method = 'S-PROG', thresholds = [0, 0.5, 1, 2, 5, 10, 30], 
                 leadtimes = [30,60,90], save_after_n_samples = 0, scales = [1, 2, 4, 16, 32, 64]):
        self.nowcast_method = nowcast_method
    
        ## Create dictionaries to compute the model errors
        # dict for each threshold and leadtime combination. shape = (n_leadtimes, n_thresholds))
        self.cat_dicts = np.array([[det_cat_fct_init(thr) for thr in thresholds] for _ in leadtimes])
        # For the MSE and MAE create dictonary per leadtime to accumulate the errors
        self.cont_dicts = np.array([det_cont_fct_init() for _ in leadtimes])
        
        
        self.fss_dicts = np.array([[[fss_init(thr = thr, scale = scale) 
                                     for scale in scales] 
                                    for thr in thresholds]
                                   for _ in leadtimes])
        
        self.leadtimes = leadtimes
        self.thresholds = thresholds
        self.scales = scales

        self.save_after_n_samples = save_after_n_samples
        self.n_verifies = 0

    def verify(self, y, y_pred, leadtime):
        index = self.leadtimes.index(leadtime)
        for cat_dict in self.cat_dicts[index]:
            det_cat_fct_accum(cat_dict, obs = y, pred = y_pred)
        det_cont_fct_accum(self.cont_dicts[index], obs = y, pred = y_pred)
        
        for i, fss_thr in enumerate(self.fss_dicts[index]):
            for j, fss_scale in enumerate(fss_thr):
                fss_accum(self.fss_dicts[index, i, j], X_o = y, X_f = y_pred)
                
        # Make checkpount if model went through n samples
        self.n_verifies +=1
        # verify is called for each lead time
        n_samples = self.n_verifies/len(self.leadtimes)
        if  self.save_after_n_samples > 0 and n_samples % self.save_after_n_samples == 0:
            self.save_accum_scores(n_samples)

    def get_scores(self):
        cat_scores = []
        # add threshold as key to the dicts
        for i, lt in enumerate(self.leadtimes):
            for j, thr in enumerate(self.thresholds):
                cat_score = det_cat_fct_compute(self.cat_dicts[i,j], scores = ['POD', 'CSI', 'FAR', 'BIAS'])
                cat_score['threshold'] = thr
                cat_score['leadtime'] = lt
                cat_scores.append(cat_score)

        cont_scores = [det_cont_fct_compute(cont_dict, scores = ['MSE', 'MAE']) for cont_dict in self.cont_dicts]
        # add lead time as key to the dicts
        for lt, cont_score in zip(self.leadtimes, cont_scores):
            cont_score['leadtime'] = lt
        return cat_scores, cont_scores

    def save_accum_scores(self, n_samples):
        np.save('results/cat_dicts_{}'.format(self.nowcast_method), self.cat_dicts)
        np.save('results/cont_dicts_{}'.format(self.nowcast_method), self.cont_dicts)
        np.save('results/fss_dicts_{}'.format(self.nowcast_method), self.fss_dicts)
        np.save('results/n_sample_{}'.format(self.nowcast_method), n_samples)

    def load_accum_scores(self):
        self.cat_dicts = np.load('results/cat_dicts_{}.npy'.format(self.nowcast_method), allow_pickle=True)
        self.cont_dicts = np.load('results/cont_dicts_{}.npy'.format(self.nowcast_method), allow_pickle=True)
        self.fss_dicts = np.load('results/fss_dicts_{}.npy'.format(self.nowcast_method), allow_pickle=True)
        self.n_verifies = 3 * np.load('results/n_sample_{}.npy'.format(self.nowcast_method))
       
    
def validate_model(model, run_name, on_test_set = False, random_split=False, resize_method = tf.image.ResizeMethod.BILINEAR):
    if not on_test_set:
        dataset = 'datasets/val2019_3y_30m.npy'
        data_name = 'val'
        if random_split:
            dataset = 'datasets/val_randomsplit.npy'
            data_name+='_randomsplit'
    else:
        dataset = 'datasets/test2020_3y_30m.npy'
        data_name = 'test'
        if random_split:
            dataset = 'datasets/test_randomsplit.npy'
            data_name+='_randomsplit'
     
    
    # First load the data 
    list_IDs = np.load(dataset, allow_pickle = True)
    
    norm_method = 'minmax'
    downscale256 = True
    convert_to_dbz = True
    y_is_rtcor = True
    
    # Create generators:
    # Create generator to load preprocessed input data
    gen = DataGenerator(list_IDs, batch_size=1, x_seq_size=6, 
                                           y_seq_size=3, norm_method = norm_method, load_prep=True,
                             downscale256 = downscale256, convert_to_dbz = convert_to_dbz, 
                                  y_is_rtcor = y_is_rtcor, shuffle=False)
    # Create a generator that loads the original target data
    cp_gen = DataGenerator(gen.list_IDs, batch_size=gen.batch_size, x_seq_size=gen.inp_shape[0], 
                                           y_seq_size=gen.out_shape[0], norm_method=None, load_prep=False,
                             downscale256 = False, convert_to_dbz = False, 
                                  y_is_rtcor = gen.y_is_rtcor, shuffle=False, crop_y=False)

    # Init evaluator object to store metrics
    save_as = run_name + '_' + data_name
    if resize_method == tf.image.ResizeMethod.NEAREST_NEIGHBOR:
        save_as = save_as + '_nearest'
    evaluator = Evaluator(save_after_n_samples = 1, nowcast_method = save_as)
    
    # Create directory to save predictions
    pred_path = config.dir_pred + save_as + '/' 
    if not os.path.exists(pred_path):
        os.makedirs(pred_path) 
    
    
    mask = get_mask_rtcor()
    
    # zip the two generators so that the preprocessed X matches the target Y data
    for (xs_prep, _), (_, ys), (_, ys_timestamps) in tqdm(zip(gen, cp_gen, list_IDs)):
        ys_pred = model.predict(xs_prep)        
        ys_pred = undo_prep(ys_pred, norm_method = norm_method, r_to_dbz=convert_to_dbz, 
                            downscale256 = downscale256, resize_method = resize_method)
        for y_pred, y_target in zip(ys_pred, ys):
            leadtimes = [30, 60, 90]
            for i, leadtime in enumerate(leadtimes):
                R_forecast = np.squeeze(np.array(y_pred[i])) * mask
                R_target = np.squeeze(np.array(y_target[i]))
                
                # save forecast:
                fn_forecast = ys_timestamps[i]+'_lt{}'.format(leadtime)
                np.save(pred_path+fn_forecast, R_forecast)
                
                # eval
                evaluator.verify(R_target, R_forecast, leadtime=leadtime)

