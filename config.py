import utils

yellow = (150, 150, 45)
utils.RGB2HSV(yellow)

lowerColor = (22, 100, 100)
upperColor = (38, 250, 250)

warmup_frame = 60
num_of_frames = 600
num_of_timestep = 10
labels = ['hold', 'pass', 'throw']

model_path = './models/simple_lstm_model_013.h5'