import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data
from PIL import Image

import sys

from collections import OrderedDict
import os
import numpy as np
import glob

import random
import pickle


all_classes = ['airplane_01_pivothead_hodgepodge',
'airplane_02_pivothead_hodgepodge',
'airplane_03_pivothead_hodgepodge',
'airplane_04_pivothead_hodgepodge',
'airplane_05_pivothead_hodgepodge',
'airplane_06_pivothead_hodgepodge',
'airplane_07_pivothead_hodgepodge',
'airplane_08_pivothead_hodgepodge',
'airplane_09_pivothead_hodgepodge',
'airplane_10_pivothead_hodgepodge',
'airplane_11_pivothead_hodgepodge',
'airplane_12_pivothead_hodgepodge',
'airplane_13_pivothead_hodgepodge',
'airplane_14_pivothead_hodgepodge',
'airplane_15_pivothead_hodgepodge',
'airplane_16_pivothead_hodgepodge',
'airplane_17_pivothead_hodgepodge',
'airplane_18_pivothead_hodgepodge',
'airplane_19_pivothead_hodgepodge',
'airplane_20_pivothead_hodgepodge',
'airplane_21_pivothead_hodgepodge',
'airplane_22_pivothead_hodgepodge',
'airplane_23_pivothead_hodgepodge',
'airplane_24_pivothead_hodgepodge',
'airplane_25_pivothead_hodgepodge',
'airplane_26_pivothead_hodgepodge',
'airplane_27_pivothead_hodgepodge',
'airplane_28_pivothead_hodgepodge',
'airplane_29_pivothead_hodgepodge',
'airplane_30_pivothead_hodgepodge',
'ball_01_pivothead_hodgepodge',
'ball_02_pivothead_hodgepodge',
'ball_03_pivothead_hodgepodge',
'ball_04_pivothead_hodgepodge',
'ball_05_pivothead_hodgepodge',
'ball_06_pivothead_hodgepodge',
'ball_07_pivothead_hodgepodge',
'ball_08_pivothead_hodgepodge',
'ball_09_pivothead_hodgepodge',
'ball_10_pivothead_hodgepodge',
'ball_11_pivothead_hodgepodge',
'ball_12_pivothead_hodgepodge',
'ball_13_pivothead_hodgepodge',
'ball_14_pivothead_hodgepodge',
'ball_15_pivothead_hodgepodge',
'ball_16_pivothead_hodgepodge',
'ball_17_pivothead_hodgepodge',
'ball_18_pivothead_hodgepodge',
'ball_19_pivothead_hodgepodge',
'ball_20_pivothead_hodgepodge',
'ball_21_pivothead_hodgepodge',
'ball_22_pivothead_hodgepodge',
'ball_23_pivothead_hodgepodge',
'ball_24_pivothead_hodgepodge',
'ball_25_pivothead_hodgepodge',
'ball_26_pivothead_hodgepodge',
'ball_27_pivothead_hodgepodge',
'ball_28_pivothead_hodgepodge',
'ball_29_pivothead_hodgepodge',
'ball_30_pivothead_hodgepodge',
'car_01_pivothead_hodgepodge',
'car_02_pivothead_hodgepodge',
'car_03_pivothead_hodgepodge',
'car_04_pivothead_hodgepodge',
'car_05_pivothead_hodgepodge',
'car_06_pivothead_hodgepodge',
'car_07_pivothead_hodgepodge',
'car_08_pivothead_hodgepodge',
'car_09_pivothead_hodgepodge',
'car_10_pivothead_hodgepodge',
'car_11_pivothead_hodgepodge',
'car_12_pivothead_hodgepodge',
'car_13_pivothead_hodgepodge',
'car_14_pivothead_hodgepodge',
'car_15_pivothead_hodgepodge',
'car_16_pivothead_hodgepodge',
'car_17_pivothead_hodgepodge',
'car_18_pivothead_hodgepodge',
'car_19_pivothead_hodgepodge',
'car_20_pivothead_hodgepodge',
'car_21_pivothead_hodgepodge',
'car_22_pivothead_hodgepodge',
'car_23_pivothead_hodgepodge',
'car_24_pivothead_hodgepodge',
'car_25_pivothead_hodgepodge',
'car_26_pivothead_hodgepodge',
'car_27_pivothead_hodgepodge',
'car_28_pivothead_hodgepodge',
'car_29_pivothead_hodgepodge',
'car_30_pivothead_hodgepodge',
'cat_01_pivothead_hodgepodge',
'cat_02_pivothead_hodgepodge',
'cat_03_pivothead_hodgepodge',
'cat_04_pivothead_hodgepodge',
'cat_05_pivothead_hodgepodge',
'cat_06_pivothead_hodgepodge',
'cat_07_pivothead_hodgepodge',
'cat_08_pivothead_hodgepodge',
'cat_09_pivothead_hodgepodge',
'cat_10_pivothead_hodgepodge',
'cat_11_pivothead_hodgepodge',
'cat_12_pivothead_hodgepodge',
'cat_13_pivothead_hodgepodge',
'cat_14_pivothead_hodgepodge',
'cat_15_pivothead_hodgepodge',
'cat_16_pivothead_hodgepodge',
'cat_17_pivothead_hodgepodge',
'cat_18_pivothead_hodgepodge',
'cat_19_pivothead_hodgepodge',
'cat_20_pivothead_hodgepodge',
'cat_21_pivothead_hodgepodge',
'cat_22_pivothead_hodgepodge',
'cat_23_pivothead_hodgepodge',
'cat_24_pivothead_hodgepodge',
'cat_25_pivothead_hodgepodge',
'cat_26_pivothead_hodgepodge',
'cat_27_pivothead_hodgepodge',
'cat_28_pivothead_hodgepodge',
'cat_29_pivothead_hodgepodge',
'cat_30_pivothead_hodgepodge',
'cup_01_pivothead_hodgepodge',
'cup_02_pivothead_hodgepodge',
'cup_03_pivothead_hodgepodge',
'cup_04_pivothead_hodgepodge',
'cup_05_pivothead_hodgepodge',
'cup_06_pivothead_hodgepodge',
'cup_07_pivothead_hodgepodge',
'cup_08_pivothead_hodgepodge',
'cup_09_pivothead_hodgepodge',
'cup_10_pivothead_hodgepodge',
'cup_11_pivothead_hodgepodge',
'cup_12_pivothead_hodgepodge',
'cup_13_pivothead_hodgepodge',
'cup_14_pivothead_hodgepodge',
'cup_15_pivothead_hodgepodge',
'cup_16_pivothead_hodgepodge',
'cup_17_pivothead_hodgepodge',
'cup_18_pivothead_hodgepodge',
'cup_19_pivothead_hodgepodge',
'cup_20_pivothead_hodgepodge',
'cup_21_pivothead_hodgepodge',
'cup_22_pivothead_hodgepodge',
'cup_23_pivothead_hodgepodge',
'cup_24_pivothead_hodgepodge',
'cup_25_pivothead_hodgepodge',
'cup_26_pivothead_hodgepodge',
'cup_27_pivothead_hodgepodge',
'cup_28_pivothead_hodgepodge',
'cup_29_pivothead_hodgepodge',
'cup_30_pivothead_hodgepodge',
'duck_01_pivothead_hodgepodge',
'duck_02_pivothead_hodgepodge',
'duck_03_pivothead_hodgepodge',
'duck_04_pivothead_hodgepodge',
'duck_05_pivothead_hodgepodge',
'duck_06_pivothead_hodgepodge',
'duck_07_pivothead_hodgepodge',
'duck_08_pivothead_hodgepodge',
'duck_09_pivothead_hodgepodge',
'duck_10_pivothead_hodgepodge',
'duck_11_pivothead_hodgepodge',
'duck_12_pivothead_hodgepodge',
'duck_13_pivothead_hodgepodge',
'duck_14_pivothead_hodgepodge',
'duck_15_pivothead_hodgepodge',
'duck_16_pivothead_hodgepodge',
'duck_17_pivothead_hodgepodge',
'duck_18_pivothead_hodgepodge',
'duck_19_pivothead_hodgepodge',
'duck_20_pivothead_hodgepodge',
'duck_21_pivothead_hodgepodge',
'duck_22_pivothead_hodgepodge',
'duck_23_pivothead_hodgepodge',
'duck_24_pivothead_hodgepodge',
'duck_25_pivothead_hodgepodge',
'duck_26_pivothead_hodgepodge',
'duck_27_pivothead_hodgepodge',
'duck_28_pivothead_hodgepodge',
'duck_29_pivothead_hodgepodge',
'duck_30_pivothead_hodgepodge',
'giraffe_01_pivothead_hodgepodge',
'giraffe_02_pivothead_hodgepodge',
'giraffe_03_pivothead_hodgepodge',
'giraffe_04_pivothead_hodgepodge',
'giraffe_05_pivothead_hodgepodge',
'giraffe_06_pivothead_hodgepodge',
'giraffe_07_pivothead_hodgepodge',
'giraffe_08_pivothead_hodgepodge',
'giraffe_09_pivothead_hodgepodge',
'giraffe_10_pivothead_hodgepodge',
'giraffe_11_pivothead_hodgepodge',
'giraffe_12_pivothead_hodgepodge',
'giraffe_13_pivothead_hodgepodge',
'giraffe_14_pivothead_hodgepodge',
'giraffe_15_pivothead_hodgepodge',
'giraffe_16_pivothead_hodgepodge',
'giraffe_17_pivothead_hodgepodge',
'giraffe_18_pivothead_hodgepodge',
'giraffe_19_pivothead_hodgepodge',
'giraffe_20_pivothead_hodgepodge',
'giraffe_21_pivothead_hodgepodge',
'giraffe_22_pivothead_hodgepodge',
'giraffe_23_pivothead_hodgepodge',
'giraffe_24_pivothead_hodgepodge',
'giraffe_25_pivothead_hodgepodge',
'giraffe_26_pivothead_hodgepodge',
'giraffe_27_pivothead_hodgepodge',
'giraffe_28_pivothead_hodgepodge',
'giraffe_29_pivothead_hodgepodge',
'giraffe_30_pivothead_hodgepodge',
'helicopter_01_pivothead_hodgepodge',
'helicopter_02_pivothead_hodgepodge',
'helicopter_03_pivothead_hodgepodge',
'helicopter_04_pivothead_hodgepodge',
'helicopter_05_pivothead_hodgepodge',
'helicopter_06_pivothead_hodgepodge',
'helicopter_07_pivothead_hodgepodge',
'helicopter_08_pivothead_hodgepodge',
'helicopter_09_pivothead_hodgepodge',
'helicopter_10_pivothead_hodgepodge',
'helicopter_11_pivothead_hodgepodge',
'helicopter_12_pivothead_hodgepodge',
'helicopter_13_pivothead_hodgepodge',
'helicopter_14_pivothead_hodgepodge',
'helicopter_15_pivothead_hodgepodge',
'helicopter_16_pivothead_hodgepodge',
'helicopter_17_pivothead_hodgepodge',
'helicopter_18_pivothead_hodgepodge',
'helicopter_19_pivothead_hodgepodge',
'helicopter_20_pivothead_hodgepodge',
'helicopter_21_pivothead_hodgepodge',
'helicopter_22_pivothead_hodgepodge',
'helicopter_23_pivothead_hodgepodge',
'helicopter_24_pivothead_hodgepodge',
'helicopter_25_pivothead_hodgepodge',
'helicopter_26_pivothead_hodgepodge',
'helicopter_27_pivothead_hodgepodge',
'helicopter_28_pivothead_hodgepodge',
'helicopter_29_pivothead_hodgepodge',
'helicopter_30_pivothead_hodgepodge',
'horse_01_pivothead_hodgepodge',
'horse_02_pivothead_hodgepodge',
'horse_03_pivothead_hodgepodge',
'horse_04_pivothead_hodgepodge',
'horse_05_pivothead_hodgepodge',
'horse_06_pivothead_hodgepodge',
'horse_07_pivothead_hodgepodge',
'horse_08_pivothead_hodgepodge',
'horse_09_pivothead_hodgepodge',
'horse_10_pivothead_hodgepodge',
'horse_11_pivothead_hodgepodge',
'horse_12_pivothead_hodgepodge',
'horse_13_pivothead_hodgepodge',
'horse_14_pivothead_hodgepodge',
'horse_15_pivothead_hodgepodge',
'horse_16_pivothead_hodgepodge',
'horse_17_pivothead_hodgepodge',
'horse_18_pivothead_hodgepodge',
'horse_19_pivothead_hodgepodge',
'horse_20_pivothead_hodgepodge',
'horse_21_pivothead_hodgepodge',
'horse_22_pivothead_hodgepodge',
'horse_23_pivothead_hodgepodge',
'horse_24_pivothead_hodgepodge',
'horse_25_pivothead_hodgepodge',
'horse_26_pivothead_hodgepodge',
'horse_27_pivothead_hodgepodge',
'horse_28_pivothead_hodgepodge',
'horse_29_pivothead_hodgepodge',
'horse_30_pivothead_hodgepodge',
'mug_01_pivothead_hodgepodge',
'mug_02_pivothead_hodgepodge',
'mug_03_pivothead_hodgepodge',
'mug_04_pivothead_hodgepodge',
'mug_05_pivothead_hodgepodge',
'mug_06_pivothead_hodgepodge',
'mug_07_pivothead_hodgepodge',
'mug_08_pivothead_hodgepodge',
'mug_09_pivothead_hodgepodge',
'mug_10_pivothead_hodgepodge',
'mug_11_pivothead_hodgepodge',
'mug_12_pivothead_hodgepodge',
'mug_13_pivothead_hodgepodge',
'mug_14_pivothead_hodgepodge',
'mug_15_pivothead_hodgepodge',
'mug_16_pivothead_hodgepodge',
'mug_17_pivothead_hodgepodge',
'mug_18_pivothead_hodgepodge',
'mug_19_pivothead_hodgepodge',
'mug_20_pivothead_hodgepodge',
'mug_21_pivothead_hodgepodge',
'mug_22_pivothead_hodgepodge',
'mug_23_pivothead_hodgepodge',
'mug_24_pivothead_hodgepodge',
'mug_25_pivothead_hodgepodge',
'mug_26_pivothead_hodgepodge',
'mug_27_pivothead_hodgepodge',
'mug_28_pivothead_hodgepodge',
'mug_29_pivothead_hodgepodge',
'mug_30_pivothead_hodgepodge',
'spoon_01_pivothead_hodgepodge',
'spoon_02_pivothead_hodgepodge',
'spoon_03_pivothead_hodgepodge',
'spoon_04_pivothead_hodgepodge',
'spoon_05_pivothead_hodgepodge',
'spoon_06_pivothead_hodgepodge',
'spoon_07_pivothead_hodgepodge',
'spoon_08_pivothead_hodgepodge',
'spoon_09_pivothead_hodgepodge',
'spoon_10_pivothead_hodgepodge',
'spoon_11_pivothead_hodgepodge',
'spoon_12_pivothead_hodgepodge',
'spoon_13_pivothead_hodgepodge',
'spoon_14_pivothead_hodgepodge',
'spoon_15_pivothead_hodgepodge',
'spoon_16_pivothead_hodgepodge',
'spoon_17_pivothead_hodgepodge',
'spoon_18_pivothead_hodgepodge',
'spoon_19_pivothead_hodgepodge',
'spoon_20_pivothead_hodgepodge',
'spoon_21_pivothead_hodgepodge',
'spoon_22_pivothead_hodgepodge',
'spoon_23_pivothead_hodgepodge',
'spoon_24_pivothead_hodgepodge',
'spoon_25_pivothead_hodgepodge',
'spoon_26_pivothead_hodgepodge',
'spoon_27_pivothead_hodgepodge',
'spoon_28_pivothead_hodgepodge',
'spoon_29_pivothead_hodgepodge',
'spoon_30_pivothead_hodgepodge',
'truck_01_pivothead_hodgepodge',
'truck_02_pivothead_hodgepodge',
'truck_03_pivothead_hodgepodge',
'truck_04_pivothead_hodgepodge',
'truck_05_pivothead_hodgepodge',
'truck_06_pivothead_hodgepodge',
'truck_07_pivothead_hodgepodge',
'truck_08_pivothead_hodgepodge',
'truck_09_pivothead_hodgepodge',
'truck_10_pivothead_hodgepodge',
'truck_11_pivothead_hodgepodge',
'truck_12_pivothead_hodgepodge',
'truck_13_pivothead_hodgepodge',
'truck_14_pivothead_hodgepodge',
'truck_15_pivothead_hodgepodge',
'truck_16_pivothead_hodgepodge',
'truck_17_pivothead_hodgepodge',
'truck_18_pivothead_hodgepodge',
'truck_19_pivothead_hodgepodge',
'truck_20_pivothead_hodgepodge',
'truck_21_pivothead_hodgepodge',
'truck_22_pivothead_hodgepodge',
'truck_23_pivothead_hodgepodge',
'truck_24_pivothead_hodgepodge',
'truck_25_pivothead_hodgepodge',
'truck_26_pivothead_hodgepodge',
'truck_27_pivothead_hodgepodge',
'truck_28_pivothead_hodgepodge',
'truck_29_pivothead_hodgepodge',
'truck_30_pivothead_hodgepodge']


known_classes = ['airplane_01_pivothead_hodgepodge', 'airplane_03_pivothead_hodgepodge',
                 'airplane_04_pivothead_hodgepodge', 'airplane_05_pivothead_hodgepodge',
                 'airplane_06_pivothead_hodgepodge', 'airplane_07_pivothead_hodgepodge',
                 'airplane_09_pivothead_hodgepodge', 'airplane_12_pivothead_hodgepodge',
                 'airplane_13_pivothead_hodgepodge', 'airplane_14_pivothead_hodgepodge',
                 'airplane_15_pivothead_hodgepodge', 'airplane_16_pivothead_hodgepodge',
                 'airplane_18_pivothead_hodgepodge', 'airplane_19_pivothead_hodgepodge',
                 'airplane_20_pivothead_hodgepodge',
                 'airplane_21_pivothead_hodgepodge',
                 'airplane_22_pivothead_hodgepodge',
                 'airplane_23_pivothead_hodgepodge',
                 'airplane_25_pivothead_hodgepodge',
                 'airplane_27_pivothead_hodgepodge',
                 'airplane_28_pivothead_hodgepodge',
                 'airplane_29_pivothead_hodgepodge',
                 'airplane_30_pivothead_hodgepodge', 'ball_01_pivothead_hodgepodge',
                 'ball_04_pivothead_hodgepodge', 'ball_06_pivothead_hodgepodge',
                 'ball_12_pivothead_hodgepodge', 'ball_13_pivothead_hodgepodge',
                 'ball_15_pivothead_hodgepodge', 'ball_16_pivothead_hodgepodge',
                 'ball_20_pivothead_hodgepodge', 'ball_21_pivothead_hodgepodge',
                 'ball_22_pivothead_hodgepodge', 'ball_23_pivothead_hodgepodge',
                 'ball_24_pivothead_hodgepodge', 'ball_26_pivothead_hodgepodge',
                 'ball_27_pivothead_hodgepodge', 'ball_28_pivothead_hodgepodge',
                 'ball_29_pivothead_hodgepodge', 'ball_30_pivothead_hodgepodge',
                 'car_01_pivothead_hodgepodge', 'car_02_pivothead_hodgepodge',
                 'car_03_pivothead_hodgepodge', 'car_06_pivothead_hodgepodge',
                 'car_08_pivothead_hodgepodge', 'car_10_pivothead_hodgepodge',
                 'car_11_pivothead_hodgepodge', 'car_12_pivothead_hodgepodge',
                 'car_14_pivothead_hodgepodge', 'car_15_pivothead_hodgepodge',
                 'car_17_pivothead_hodgepodge', 'car_18_pivothead_hodgepodge',
                 'car_19_pivothead_hodgepodge', 'car_22_pivothead_hodgepodge',
                 'car_24_pivothead_hodgepodge', 'car_27_pivothead_hodgepodge',
                 'car_28_pivothead_hodgepodge', 'car_30_pivothead_hodgepodge',
                 'cat_01_pivothead_hodgepodge', 'cat_02_pivothead_hodgepodge',
                 'cat_04_pivothead_hodgepodge', 'cat_06_pivothead_hodgepodge',
                 'cat_07_pivothead_hodgepodge', 'cat_08_pivothead_hodgepodge',
                 'cat_09_pivothead_hodgepodge', 'cat_10_pivothead_hodgepodge',
                 'cat_11_pivothead_hodgepodge', 'cat_12_pivothead_hodgepodge',
                 'cat_15_pivothead_hodgepodge', 'cat_16_pivothead_hodgepodge',
                 'cat_19_pivothead_hodgepodge', 'cat_20_pivothead_hodgepodge',
                 'cat_22_pivothead_hodgepodge', 'cat_23_pivothead_hodgepodge',
                 'cat_25_pivothead_hodgepodge', 'cat_26_pivothead_hodgepodge',
                 'cat_28_pivothead_hodgepodge', 'cat_29_pivothead_hodgepodge',
                 'cup_03_pivothead_hodgepodge', 'cup_04_pivothead_hodgepodge',
                 'cup_05_pivothead_hodgepodge', 'cup_06_pivothead_hodgepodge',
                 'cup_07_pivothead_hodgepodge', 'cup_08_pivothead_hodgepodge',
                 'cup_10_pivothead_hodgepodge', 'cup_11_pivothead_hodgepodge',
                 'cup_13_pivothead_hodgepodge', 'cup_14_pivothead_hodgepodge',
                 'cup_16_pivothead_hodgepodge', 'cup_21_pivothead_hodgepodge',
                 'cup_24_pivothead_hodgepodge', 'cup_26_pivothead_hodgepodge',
                 'duck_02_pivothead_hodgepodge', 'duck_04_pivothead_hodgepodge',
                 'duck_05_pivothead_hodgepodge', 'duck_06_pivothead_hodgepodge',
                 'duck_07_pivothead_hodgepodge', 'duck_09_pivothead_hodgepodge',
                 'duck_10_pivothead_hodgepodge', 'duck_11_pivothead_hodgepodge',
                 'duck_12_pivothead_hodgepodge', 'duck_13_pivothead_hodgepodge',
                 'duck_15_pivothead_hodgepodge', 'duck_16_pivothead_hodgepodge',
                 'duck_18_pivothead_hodgepodge', 'duck_19_pivothead_hodgepodge',
                 'duck_20_pivothead_hodgepodge', 'duck_21_pivothead_hodgepodge',
                 'duck_22_pivothead_hodgepodge', 'duck_23_pivothead_hodgepodge',
                 'duck_26_pivothead_hodgepodge', 'duck_27_pivothead_hodgepodge',
                 'duck_29_pivothead_hodgepodge', 'duck_30_pivothead_hodgepodge',
                 'giraffe_01_pivothead_hodgepodge',
                 'giraffe_04_pivothead_hodgepodge',
                 'giraffe_07_pivothead_hodgepodge',
                 'giraffe_08_pivothead_hodgepodge',
                 'giraffe_10_pivothead_hodgepodge',
                 'giraffe_11_pivothead_hodgepodge',
                 'giraffe_12_pivothead_hodgepodge',
                 'giraffe_13_pivothead_hodgepodge',
                 'giraffe_14_pivothead_hodgepodge',
                 'giraffe_16_pivothead_hodgepodge',
                 'giraffe_17_pivothead_hodgepodge',
                 'giraffe_18_pivothead_hodgepodge',
                 'giraffe_21_pivothead_hodgepodge',
                 'giraffe_24_pivothead_hodgepodge',
                 'giraffe_25_pivothead_hodgepodge',
                 'giraffe_26_pivothead_hodgepodge',
                 'giraffe_27_pivothead_hodgepodge',
                 'giraffe_28_pivothead_hodgepodge',
                 'giraffe_29_pivothead_hodgepodge',
                 'helicopter_03_pivothead_hodgepodge',
                 'helicopter_04_pivothead_hodgepodge',
                 'helicopter_05_pivothead_hodgepodge',
                 'helicopter_06_pivothead_hodgepodge',
                 'helicopter_07_pivothead_hodgepodge',
                 'helicopter_08_pivothead_hodgepodge',
                 'helicopter_09_pivothead_hodgepodge',
                 'helicopter_10_pivothead_hodgepodge',
                 'helicopter_12_pivothead_hodgepodge',
                 'helicopter_14_pivothead_hodgepodge',
                 'helicopter_15_pivothead_hodgepodge',
                 'helicopter_16_pivothead_hodgepodge',
                 'helicopter_17_pivothead_hodgepodge',
                 'helicopter_20_pivothead_hodgepodge',
                 'helicopter_23_pivothead_hodgepodge',
                 'helicopter_24_pivothead_hodgepodge',
                 'helicopter_25_pivothead_hodgepodge',
                 'helicopter_26_pivothead_hodgepodge',
                 'helicopter_29_pivothead_hodgepodge',
                 'helicopter_30_pivothead_hodgepodge',
                 'horse_04_pivothead_hodgepodge', 'horse_06_pivothead_hodgepodge',
                 'horse_10_pivothead_hodgepodge', 'horse_11_pivothead_hodgepodge',
                 'horse_12_pivothead_hodgepodge', 'horse_13_pivothead_hodgepodge',
                 'horse_17_pivothead_hodgepodge', 'horse_18_pivothead_hodgepodge',
                 'horse_19_pivothead_hodgepodge', 'horse_20_pivothead_hodgepodge',
                 'horse_21_pivothead_hodgepodge', 'horse_22_pivothead_hodgepodge',
                 'horse_23_pivothead_hodgepodge', 'horse_24_pivothead_hodgepodge',
                 'horse_27_pivothead_hodgepodge', 'horse_28_pivothead_hodgepodge',
                 'horse_29_pivothead_hodgepodge', 'horse_30_pivothead_hodgepodge',
                 'mug_01_pivothead_hodgepodge', 'mug_03_pivothead_hodgepodge',
                 'mug_05_pivothead_hodgepodge', 'mug_06_pivothead_hodgepodge',
                 'mug_08_pivothead_hodgepodge', 'mug_09_pivothead_hodgepodge',
                 'mug_12_pivothead_hodgepodge', 'mug_14_pivothead_hodgepodge',
                 'mug_18_pivothead_hodgepodge', 'mug_20_pivothead_hodgepodge',
                 'mug_21_pivothead_hodgepodge', 'mug_22_pivothead_hodgepodge',
                 'mug_23_pivothead_hodgepodge', 'mug_24_pivothead_hodgepodge',
                 'mug_26_pivothead_hodgepodge', 'mug_27_pivothead_hodgepodge',
                 'mug_28_pivothead_hodgepodge', 'mug_29_pivothead_hodgepodge',
                 'spoon_01_pivothead_hodgepodge', 'spoon_02_pivothead_hodgepodge',
                 'spoon_03_pivothead_hodgepodge', 'spoon_04_pivothead_hodgepodge',
                 'spoon_06_pivothead_hodgepodge', 'spoon_07_pivothead_hodgepodge',
                 'spoon_08_pivothead_hodgepodge', 'spoon_09_pivothead_hodgepodge',
                 'spoon_10_pivothead_hodgepodge', 'spoon_11_pivothead_hodgepodge',
                 'spoon_12_pivothead_hodgepodge', 'spoon_13_pivothead_hodgepodge',
                 'spoon_14_pivothead_hodgepodge', 'spoon_15_pivothead_hodgepodge',
                 'spoon_16_pivothead_hodgepodge', 'spoon_17_pivothead_hodgepodge',
                 'spoon_18_pivothead_hodgepodge', 'spoon_19_pivothead_hodgepodge',
                 'spoon_20_pivothead_hodgepodge', 'spoon_21_pivothead_hodgepodge',
                 'spoon_22_pivothead_hodgepodge', 'spoon_24_pivothead_hodgepodge',
                 'spoon_26_pivothead_hodgepodge', 'spoon_27_pivothead_hodgepodge',
                 'spoon_28_pivothead_hodgepodge', 'spoon_30_pivothead_hodgepodge',
                 'truck_01_pivothead_hodgepodge', 'truck_02_pivothead_hodgepodge',
                 'truck_03_pivothead_hodgepodge', 'truck_04_pivothead_hodgepodge',
                 'truck_05_pivothead_hodgepodge', 'truck_06_pivothead_hodgepodge',
                 'truck_07_pivothead_hodgepodge', 'truck_09_pivothead_hodgepodge',
                 'truck_10_pivothead_hodgepodge', 'truck_11_pivothead_hodgepodge',
                 'truck_12_pivothead_hodgepodge', 'truck_13_pivothead_hodgepodge',
                 'truck_15_pivothead_hodgepodge', 'truck_16_pivothead_hodgepodge',
                 'truck_17_pivothead_hodgepodge', 'truck_19_pivothead_hodgepodge',
                 'truck_21_pivothead_hodgepodge', 'truck_22_pivothead_hodgepodge',
                 'truck_23_pivothead_hodgepodge', 'truck_24_pivothead_hodgepodge',
                 'truck_25_pivothead_hodgepodge', 'truck_26_pivothead_hodgepodge',
                 'truck_27_pivothead_hodgepodge', 'truck_29_pivothead_hodgepodge',
                 'truck_30_pivothead_hodgepodge']


def ordered_glob(rootdir='.', suffix='', class_id=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    filenames = []
    filenames_folder = []

    folders = glob.glob(rootdir + "/*")

    for folder in folders:
        folder_id = os.path.split(folder)[1]

        if folder_id in known_classes:
            folder_path = folder + "/*"

            filenames_folder = glob.glob(folder_path)
            filenames_folder.sort()
            filenames.extend(filenames_folder)

    return filenames


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


def get_different_object(filename):
    obj_root = os.path.split(os.path.split(filename)[0])[0]

    similar_object_group = known_classes[:]  # os.listdir(obj_root)

    obj_class = os.path.split(os.path.split(filename)[0])[1]

    obj_nxt_index = random.randint(0, len(similar_object_group) - 1)

    obj_next = similar_object_group.pop(obj_nxt_index)

    if obj_next == obj_class: obj_next = similar_object_group.pop(obj_nxt_index - 1)

    obj_next_path = os.path.join(obj_root, obj_next)

    obj_next_views = os.listdir(obj_next_path)

    obj_next_view = random.choice(obj_next_views)

    return os.path.join(obj_next_path, obj_next_view)


def get_different_view(filename):
    obj_folder = os.path.split(filename)[0]

    obj_root = os.path.split(obj_folder)

    obj_item_candidates = os.path.join(obj_root[0], obj_root[1])

    obj_views_candidates = os.listdir(obj_item_candidates)

    random_index = random.randint(0, len(obj_views_candidates) - 1)

    next_view = obj_views_candidates.pop(random_index)

    new_filename = os.path.join(obj_item_candidates, next_view)

    return new_filename


def get_nearby_view(filename):
    nby_index = 2

    vecinity = range(-nby_index, nby_index)

    obj_folder = os.path.split(filename)[0]

    obj_id = os.path.split(filename)[1]

    obj_view_candidates = os.listdir(obj_folder)

    obj_view_candidates.sort()

    obj_idx_array = obj_view_candidates.index(obj_id)

    next_view = random.choice(vecinity)

    next_view_idx = np.clip(obj_idx_array + next_view, 0, len(obj_view_candidates) - 1)

    next_view_id = obj_view_candidates[next_view_idx]

    # '/media/mikelf/media_rob/core50_v3/train_full/obj_01_scene_04/C_04_01_236.png'

    new_filename = os.path.join(obj_folder, next_view_id)

    return new_filename

def read_labels(name ):
    with open( name , 'rb') as f:
        return pickle.load(f)

labels = read_labels("/media/alexa/DATA/Datasets/toybox/labels.pkl")


class triplet_resnet_toybox(data.Dataset):
    """tless loader
    """

    def __init__(self, root, split="train", is_transform=False,
                 img_size=(224, 224), augmentations=None):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 360
        self.n_channels = 3
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}

        os.path.join(self.root, self.split)

        self.images_base = os.path.join(self.root, self.split)

        self.files[split] = ordered_glob(rootdir=self.images_base, suffix='.png')

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        img = Image.open(img_path)



        if self.split[0:5] == "train":

            img_path_similar = get_different_view(img_path)
      

            img_path_different = get_different_object(img_path)
   

            img_pos = Image.open(img_path_similar)
     

            img_neg = Image.open(img_path_different)
   


        else:

            img_next = Image.open(img_path)
            # /media/mikelf/media_rob/core50_v3/test/obj_01_scene_03/C_03_01_001.png



        img = self.resize_keepRatio(img)
        img_pos = self.resize_keepRatio(img_pos)
        img_neg = self.resize_keepRatio(img_neg)

        img = np.array(img, dtype=np.uint8)
        img_pos = np.array(img_pos, dtype=np.uint8)
        img_neg = np.array(img_neg, dtype=np.uint8)



        if self.is_transform:
            img, img_pos, img_neg = self.transform(img, img_pos, img_neg )
         

        return img, img_pos, img_neg, img_path

    def resize_keepRatio(self, img):

        old_size = img.size

        ratio = float(self.img_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (self.img_size[0], self.img_size[1]))
        new_im.paste(img, ((self.img_size[0]-new_size[0])//2,
                    (self.img_size[1]-new_size[1])//2))

        return new_im

    def transform(self, img, img_pos, img_neg):
        """transform

        :param img:
        :param lbl:
        """
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)


        img_pos = img_pos[:, :, ::-1]
        img_pos = img_pos.astype(np.float64)
        img_pos -= self.mean
        img_pos = m.imresize(img_pos, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img_pos = img_pos.astype(float) / 255.0
        # NHWC -> NCWH
        img_pos = img_pos.transpose(2, 0, 1)



        img_neg = img_neg[:, :, ::-1]
        img_neg = img_neg.astype(np.float64)
        img_neg -= self.mean
        img_neg = m.imresize(img_neg, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img_neg = img_neg.astype(float) / 255.0
        # NHWC -> NCWH
        img_neg = img_neg.transpose(2, 0, 1)




        img = torch.from_numpy(img).float()
      
        img_pos = torch.from_numpy(img_pos).float()
      
        img_neg = torch.from_numpy(img_neg).float()
     

        return img, img_pos, img_neg


if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    local_path = '/media/mikelf/media_rob/Datasets/emmi'
    dst = triplet_ae_toybox_softmax(local_path, split="train", is_transform=True, augmentations=None)
    bs = 2
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)

    f, axarr = plt.subplots(bs, 6)

    for i, data in enumerate(trainloader):
        imgs, imgs_nby, imgs_pos, imgs_pos_nby, imgs_neg, imgs_neg_nby, filenames, lbl, lbl_pos, lbl_neg = data

        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])

        imgs_nby = imgs_nby.numpy()[:, ::-1, :, :]
        imgs_nby = np.transpose(imgs_nby, [0, 2, 3, 1])

        imgs_pos = imgs_pos.numpy()[:, ::-1, :, :]
        imgs_pos = np.transpose(imgs_pos, [0, 2, 3, 1])

        imgs_pos_nby = imgs_pos_nby.numpy()[:, ::-1, :, :]
        imgs_pos_nby = np.transpose(imgs_pos_nby, [0, 2, 3, 1])

        imgs_neg = imgs_neg.numpy()[:, ::-1, :, :]
        imgs_neg = np.transpose(imgs_neg, [0, 2, 3, 1])

        imgs_neg_nby = imgs_neg_nby.numpy()[:, ::-1, :, :]
        imgs_neg_nby = np.transpose(imgs_neg_nby, [0, 2, 3, 1])

        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(imgs_nby[j])
            axarr[j][2].imshow(imgs_pos[j])
            axarr[j][3].imshow(imgs_pos_nby[j])
            axarr[j][4].imshow(imgs_neg[j])
            axarr[j][5].imshow(imgs_neg_nby[j])
            print(lbl[j], lbl_pos[j], lbl_neg[j])

            # print(filenames)

        plt.pause(0.1)
        plt.cla()

