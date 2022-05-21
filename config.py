import os
import utils
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = 'simple_lstm_model_022.h5'

yellow = (150, 150, 45)

yellow = (176, 170, 96)
red = (155, 0, 21)

white = (190, 190, 179)
white = (231, 233, 277)
utils.RGB2HSV(red)

# red
lowerColor = (0, 130, 200)
upperColor = (20, 200, 255)

# Orange
# lowerColor = (7, 130, 200)
# upperColor = (20, 200, 255)

# Yellow
# lowerColor = (22, 100, 100)
# upperColor = (38, 250, 250)

# white
# lowerColor = (0, 0, 180)
# upperColor = (255, 40, 255)

# RGB
lowerColor_RGB = (190, 110, 110)
upperColor_RGB = (255, 200, 210)

# YUV
lowerColor_YUV = (80, 120, 130)
upperColor_YUV = (150, 160, 170)

warmup_frame = 60
num_of_frames = 1000
num_of_timestep = 10
# labels = ['dribble', 'shoot', 'pass', 'hold', 'no_ball']
labels = ['heading', 'shoot', 'dribble', 'trapping', 'noball']

model_path = os.path.join(BASE_DIR, 'models', MODEL_FILE)
