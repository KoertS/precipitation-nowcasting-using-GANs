from pysteps.verification.detcatscores import det_cat_fct_init, det_cat_fct_accum, det_cat_fct_compute, det_cat_fct_merge
from pysteps.verification.detcontscores import det_cont_fct_init, det_cont_fct_accum, det_cont_fct_compute
import numpy as np
from batchcreator import DataGenerator, undo_prep
from model_builder import GAN
import tensorflow as tf

class Evaluator:
    '''
    This functions is used to validate a models predictions.
    Categorical scores and continues errors are accumulated in dictionaries.
    When validation is finished, the metrics can be computed by using these dictionaries.
    nowcast_method: indicates with what model the nowcast were made
    thresholds: precipitation threshold to use for the categorical scores
    leadtimes: the leadtimes of the predictions
    save_after_n_sample: if higher than 0, the evaluator will save its dictionary after it has seen n samples. 
    '''
    def __init__(self, nowcast_method = 'S-PROG', thresholds = [0.5, 5, 10, 30], leadtimes = [30,60,90], save_after_n_samples = 0):
        self.nowcast_method = nowcast_method
    
        ## Create dictionaries to compute the model errors
        # dict for each threshold and leadtime combination. shape = (n_leadtimes, n_thresholds))
        self.cat_dicts = np.array([[det_cat_fct_init(thr) for thr in thresholds] for _ in leadtimes])
        # For the MSE and MAE create dictonary per leadtime to accumulate the errors
        self.cont_dicts = np.array([det_cont_fct_init() for _ in leadtimes])
    
        self.leadtimes = leadtimes
        self.thresholds = thresholds

        self.save_after_n_samples = save_after_n_samples
        self.n_verifies = 0

    def verify(self, y, y_pred, leadtime):
        index = self.leadtimes.index(leadtime)
        for cat_dict in self.cat_dicts[index]:
            det_cat_fct_accum(cat_dict, y, y_pred)
        det_cont_fct_accum(self.cont_dicts[index], y, y_pred)

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
        np.save('results/n_sample_{}'.format(self.nowcast_method), n_samples)

    def load_accum_scores(self):
        self.cat_dicts = np.load('results/cat_dicts_{}.npy'.format(self.nowcast_method), allow_pickle=True)
        self.cont_dicts = np.load('results/cont_dicts_{}.npy'.format(self.nowcast_method), allow_pickle=True)
        self.n_verifies = 3 * np.load('results/n_sample_{}.npy'.format(self.nowcast_method))
       
    
def validate_model(model_path='saved_models/generator_pious_meadow_514', on_test_set = False):
    if not on_test_set:
        dataset = 'datasets/val2019_3y_30m.npy'
        data_name = 'val'
    else:
        dataset = 'datasets/test2020_3y_30m.npy'
        data_name = 'test'
        
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
    # Load the model 
    model = tf.saved_model.load(model_path)
    model_name = model_path.replace('saved_models/','')
    
    # Init evaluator object to store metrics
    save_as = model_name + '_' + data_name
    evaluator = Evaluator(save_after_n_samples = 1, nowcast_method = save_as)

    # zip the two generators so that the preprocessed X matches the target Y data
    for (xs_prep, ys_prep), (_, ys) in tqdm(zip(gen, cp_gen)):

        ys_pred = model.predict(xs_prep)        
        ys_pred = undo_prep(ys_pred, norm_method = norm_method, r_to_dbz=convert_to_dbz, 
                            downscale256 = downscale256)

        for y_pred, y_target in zip(ys_pred, ys):
            leadtimes = [30, 60, 90]
            for i, leadtime in enumerate(leadtimes):
                R_forecast = np.array(y_pred[i])
                R_target = np.array(y_target[i])
                evaluator.verify(R_target, R_forecast, leadtime=leadtime)

