from batchcreator import DataGenerator, get_list_IDs
from datetime import datetime
from model_builder import build_generator, GAN


model = GAN(rnn_type='GRU')
#model.build((None,5, 768, 700, 1))
model.compile()
model.summary()


batch_size = 1

# Create list of IDs to retrieve
x_seq_size=5
y_seq_size=1

start_dt = datetime(2019,6,4,0,0)
end_dt =  datetime(2019,6,6,0,30)
# Exclude samples were there is no rain in the input
filter_no_rain = True
list_IDs = get_list_IDs(start_dt, end_dt, x_seq_size, y_seq_size, filter_no_rain = filter_no_rain)
print(len(list_IDs))

generator = DataGenerator(list_IDs, batch_size=batch_size, x_seq_size=x_seq_size, y_seq_size=y_seq_size, norm_method='minmax')
history = model.fit(generator, epochs=1)


with open(config.path_data + 'trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
