from validation import validate_model
from model_builder import GAN, build_discriminator
import tensorflow as tf

# run_name = 'model_sunny_wind_549'
# run_name = 'model_earthy_mountain_541'

#run_name = 'model_sage_blaze_543'


# NAG model:
#run_name = 'model_quiet_yogurt_535'
# GAN model:
#run_name = 'model_silver_field_559'
# GAN model LSTM:
run_name = 'model_easy_vortex_560'

path_checkpoint = 'saved_models/' + run_name
on_test_set = False
random_split = False 

rnn_type = 'LSTM'
x_length = 6
y_length = 3
resize_method = tf.image.ResizeMethod.BILINEAR
#resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR

print('Validate {}, on_test = {}'.format(run_name, on_test_set)) 

model = GAN(rnn_type=rnn_type, x_length=x_length, 
            y_length=y_length, architecture='AENN', relu_alpha=.2,
           l_adv = 0.003, l_rec = 1, g_cycles=3, label_smoothing=0.2
            , norm_method = 'minmax', downscale256 = True, rec_with_mae= False,
           r_to_dbz = True, batch_norm = False)

# model.discriminator_seq = build_discriminator(y_length=y_length, 
#                                                      relu_alpha=.2,
#                                                     architecture='AENN', 
#                                                      downscale256 = True, batch_norm = False, drop_out = False)

# if gan model:
model.load_weights(path_checkpoint)

model_gen = model.generator

# if generator only use the following:
#model_gen.load_weights(path_checkpoint)


validate_model(model_gen, run_name, on_test_set, random_split, resize_method = resize_method)