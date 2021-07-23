from validation import validate_model
from model_builder import GAN

run_name = 'model_quiet_yogurt_535'

path_checkpoint = 'saved_models/' + run_name
on_test_set = False


x_length = 6
y_length = 3

model = GAN(rnn_type='GRU', x_length=x_length, 
            y_length=y_length, architecture='AENN', relu_alpha=.2,
           l_adv = 0.006, l_rec = 1, g_cycles=3, label_smoothing=0.2
            , norm_method = 'minmax', downscale256 = True, rec_with_mae= False,
           r_to_dbz = True, batch_norm = False)
model_gen = model.generator

model_gen.load_weights('saved_models/model_quiet_yogurt_535')


validate_model(model_gen, run_name, on_test_set)