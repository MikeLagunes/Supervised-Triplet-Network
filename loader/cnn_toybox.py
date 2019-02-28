import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data
from PIL import Image

import sys
import random

from collections import OrderedDict
import os
import numpy as np
import glob
import pickle

known_classes_test = ['airplane_01_pivothead','airplane_03_pivothead','airplane_04_pivothead','airplane_05_pivothead',
       'airplane_06_pivothead','airplane_07_pivothead','airplane_09_pivothead','airplane_12_pivothead',
       'airplane_13_pivothead','airplane_14_pivothead','airplane_15_pivothead','airplane_16_pivothead',
       'airplane_18_pivothead','airplane_19_pivothead','airplane_20_pivothead',
       'airplane_21_pivothead',
       'airplane_22_pivothead',
       'airplane_23_pivothead',
       'airplane_25_pivothead',
       'airplane_27_pivothead',
       'airplane_28_pivothead',
       'airplane_29_pivothead',
       'airplane_30_pivothead', 'ball_01_pivothead',
       'ball_04_pivothead', 'ball_06_pivothead',
       'ball_12_pivothead', 'ball_13_pivothead',
       'ball_15_pivothead', 'ball_16_pivothead',
       'ball_20_pivothead', 'ball_21_pivothead',
       'ball_22_pivothead', 'ball_23_pivothead',
       'ball_24_pivothead', 'ball_26_pivothead',
       'ball_27_pivothead', 'ball_28_pivothead',
       'ball_29_pivothead', 'ball_30_pivothead',
       'car_01_pivothead', 'car_02_pivothead',
       'car_03_pivothead', 'car_06_pivothead',
       'car_08_pivothead', 'car_10_pivothead',
       'car_11_pivothead', 'car_12_pivothead',
       'car_14_pivothead', 'car_15_pivothead',
       'car_17_pivothead', 'car_18_pivothead',
       'car_19_pivothead', 'car_22_pivothead',
       'car_24_pivothead', 'car_27_pivothead',
       'car_28_pivothead', 'car_30_pivothead',
       'cat_01_pivothead', 'cat_02_pivothead',
       'cat_04_pivothead', 'cat_06_pivothead',
       'cat_07_pivothead', 'cat_08_pivothead',
       'cat_09_pivothead', 'cat_10_pivothead',
       'cat_11_pivothead', 'cat_12_pivothead',
       'cat_15_pivothead', 'cat_16_pivothead',
       'cat_19_pivothead', 'cat_20_pivothead',
       'cat_22_pivothead', 'cat_23_pivothead',
       'cat_25_pivothead', 'cat_26_pivothead',
       'cat_28_pivothead', 'cat_29_pivothead',
       'cup_03_pivothead', 'cup_04_pivothead',
       'cup_05_pivothead', 'cup_06_pivothead',
       'cup_07_pivothead', 'cup_08_pivothead',
       'cup_10_pivothead', 'cup_11_pivothead',
       'cup_13_pivothead', 'cup_14_pivothead',
       'cup_16_pivothead', 'cup_21_pivothead',
       'cup_24_pivothead', 'cup_26_pivothead',
       'duck_02_pivothead', 'duck_04_pivothead',
       'duck_05_pivothead', 'duck_06_pivothead',
       'duck_07_pivothead', 'duck_09_pivothead',
       'duck_10_pivothead', 'duck_11_pivothead',
       'duck_12_pivothead', 'duck_13_pivothead',
       'duck_15_pivothead', 'duck_16_pivothead',
       'duck_18_pivothead', 'duck_19_pivothead',
       'duck_20_pivothead', 'duck_21_pivothead',
       'duck_22_pivothead', 'duck_23_pivothead',
       'duck_26_pivothead', 'duck_27_pivothead',
       'duck_29_pivothead', 'duck_30_pivothead',
       'giraffe_01_pivothead',
       'giraffe_04_pivothead',
       'giraffe_07_pivothead',
       'giraffe_08_pivothead',
       'giraffe_10_pivothead',
       'giraffe_11_pivothead',
       'giraffe_12_pivothead',
       'giraffe_13_pivothead',
       'giraffe_14_pivothead',
       'giraffe_16_pivothead',
       'giraffe_17_pivothead',
       'giraffe_18_pivothead',
       'giraffe_21_pivothead',
       'giraffe_24_pivothead',
       'giraffe_25_pivothead',
       'giraffe_26_pivothead',
       'giraffe_27_pivothead',
       'giraffe_28_pivothead',
       'giraffe_29_pivothead',
       'helicopter_03_pivothead',
       'helicopter_04_pivothead',
       'helicopter_05_pivothead',
       'helicopter_06_pivothead',
       'helicopter_07_pivothead',
       'helicopter_08_pivothead',
       'helicopter_09_pivothead',
       'helicopter_10_pivothead',
       'helicopter_12_pivothead',
       'helicopter_14_pivothead',
       'helicopter_15_pivothead',
       'helicopter_16_pivothead',
       'helicopter_17_pivothead',
       'helicopter_20_pivothead',
       'helicopter_23_pivothead',
       'helicopter_24_pivothead',
       'helicopter_25_pivothead',
       'helicopter_26_pivothead',
       'helicopter_29_pivothead',
       'helicopter_30_pivothead',
       'horse_04_pivothead', 'horse_06_pivothead',
       'horse_10_pivothead', 'horse_11_pivothead',
       'horse_12_pivothead', 'horse_13_pivothead',
       'horse_17_pivothead', 'horse_18_pivothead',
       'horse_19_pivothead', 'horse_20_pivothead',
       'horse_21_pivothead', 'horse_22_pivothead',
       'horse_23_pivothead', 'horse_24_pivothead',
       'horse_27_pivothead', 'horse_28_pivothead',
       'horse_29_pivothead', 'horse_30_pivothead',
       'mug_01_pivothead', 'mug_03_pivothead',
       'mug_05_pivothead', 'mug_06_pivothead',
       'mug_08_pivothead', 'mug_09_pivothead',
       'mug_12_pivothead', 'mug_14_pivothead',
       'mug_18_pivothead', 'mug_20_pivothead',
       'mug_21_pivothead', 'mug_22_pivothead',
       'mug_23_pivothead', 'mug_24_pivothead',
       'mug_26_pivothead', 'mug_27_pivothead',
       'mug_28_pivothead', 'mug_29_pivothead',
       'spoon_01_pivothead', 'spoon_02_pivothead',
       'spoon_03_pivothead', 'spoon_04_pivothead',
       'spoon_06_pivothead', 'spoon_07_pivothead',
       'spoon_08_pivothead', 'spoon_09_pivothead',
       'spoon_10_pivothead', 'spoon_11_pivothead',
       'spoon_12_pivothead', 'spoon_13_pivothead',
       'spoon_14_pivothead', 'spoon_15_pivothead',
       'spoon_16_pivothead', 'spoon_17_pivothead',
       'spoon_18_pivothead', 'spoon_19_pivothead',
       'spoon_20_pivothead', 'spoon_21_pivothead',
       'spoon_22_pivothead', 'spoon_24_pivothead',
       'spoon_26_pivothead', 'spoon_27_pivothead',
       'spoon_28_pivothead', 'spoon_30_pivothead',
       'truck_01_pivothead', 'truck_02_pivothead',
       'truck_03_pivothead', 'truck_04_pivothead',
       'truck_05_pivothead', 'truck_06_pivothead',
       'truck_07_pivothead', 'truck_09_pivothead',
       'truck_10_pivothead', 'truck_11_pivothead',
       'truck_12_pivothead', 'truck_13_pivothead',
       'truck_15_pivothead', 'truck_16_pivothead',
       'truck_17_pivothead', 'truck_19_pivothead',
       'truck_21_pivothead', 'truck_22_pivothead',
       'truck_23_pivothead', 'truck_24_pivothead',
       'truck_25_pivothead', 'truck_26_pivothead',
       'truck_27_pivothead', 'truck_29_pivothead',
       'truck_30_pivothead']

known_classes = ['airplane_01_pivothead_hodgepodge','airplane_03_pivothead_hodgepodge','airplane_04_pivothead_hodgepodge','airplane_05_pivothead_hodgepodge',
       'airplane_06_pivothead_hodgepodge','airplane_07_pivothead_hodgepodge','airplane_09_pivothead_hodgepodge','airplane_12_pivothead_hodgepodge',
       'airplane_13_pivothead_hodgepodge','airplane_14_pivothead_hodgepodge','airplane_15_pivothead_hodgepodge','airplane_16_pivothead_hodgepodge',
       'airplane_18_pivothead_hodgepodge','airplane_19_pivothead_hodgepodge','airplane_20_pivothead_hodgepodge',
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

labels_test={
'airplane_01_pivothead':0,
'airplane_02_pivothead':1,
'airplane_03_pivothead':2,
'airplane_04_pivothead':3,
'airplane_05_pivothead':4,
'airplane_06_pivothead':5,
'airplane_07_pivothead':6,
'airplane_08_pivothead':7,
'airplane_09_pivothead':8,
'airplane_10_pivothead':9,
'airplane_11_pivothead':10,
'airplane_12_pivothead':11,
'airplane_13_pivothead':12,
'airplane_14_pivothead':13,
'airplane_15_pivothead':14,
'airplane_16_pivothead':15,
'airplane_17_pivothead':16,
'airplane_18_pivothead':17,
'airplane_19_pivothead':18,
'airplane_20_pivothead':19,
'airplane_21_pivothead':20,
'airplane_22_pivothead':21,
'airplane_23_pivothead':22,
'airplane_24_pivothead':23,
'airplane_25_pivothead':24,
'airplane_26_pivothead':25,
'airplane_27_pivothead':26,
'airplane_28_pivothead':27,
'airplane_29_pivothead':28,
'airplane_30_pivothead':29,
'ball_01_pivothead':30,
'ball_02_pivothead':31,
'ball_03_pivothead':32,
'ball_04_pivothead':33,
'ball_05_pivothead':34,
'ball_06_pivothead':35,
'ball_07_pivothead':36,
'ball_08_pivothead':37,
'ball_09_pivothead':38,
'ball_10_pivothead':39,
'ball_11_pivothead':40,
'ball_12_pivothead':41,
'ball_13_pivothead':42,
'ball_14_pivothead':43,
'ball_15_pivothead':44,
'ball_16_pivothead':45,
'ball_17_pivothead':46,
'ball_18_pivothead':47,
'ball_19_pivothead':48,
'ball_20_pivothead':49,
'ball_21_pivothead':50,
'ball_22_pivothead':51,
'ball_23_pivothead':52,
'ball_24_pivothead':53,
'ball_25_pivothead':54,
'ball_26_pivothead':55,
'ball_27_pivothead':56,
'ball_28_pivothead':57,
'ball_29_pivothead':58,
'ball_30_pivothead':59,
'car_01_pivothead':60,
'car_02_pivothead':61,
'car_03_pivothead':62,
'car_04_pivothead':63,
'car_05_pivothead':64,
'car_06_pivothead':65,
'car_07_pivothead':66,
'car_08_pivothead':67,
'car_09_pivothead':68,
'car_10_pivothead':69,
'car_11_pivothead':70,
'car_12_pivothead':71,
'car_13_pivothead':72,
'car_14_pivothead':73,
'car_15_pivothead':74,
'car_16_pivothead':75,
'car_17_pivothead':76,
'car_18_pivothead':77,
'car_19_pivothead':78,
'car_20_pivothead':79,
'car_21_pivothead':80,
'car_22_pivothead':81,
'car_23_pivothead':82,
'car_24_pivothead':83,
'car_25_pivothead':84,
'car_26_pivothead':85,
'car_27_pivothead':86,
'car_28_pivothead':87,
'car_29_pivothead':88,
'car_30_pivothead':89,
'cat_01_pivothead':90,
'cat_02_pivothead':91,
'cat_03_pivothead':92,
'cat_04_pivothead':93,
'cat_05_pivothead':94,
'cat_06_pivothead':95,
'cat_07_pivothead':96,
'cat_08_pivothead':97,
'cat_09_pivothead':98,
'cat_10_pivothead':99,
'cat_11_pivothead':100,
'cat_12_pivothead':101,
'cat_13_pivothead':102,
'cat_14_pivothead':103,
'cat_15_pivothead':104,
'cat_16_pivothead':105,
'cat_17_pivothead':106,
'cat_18_pivothead':107,
'cat_19_pivothead':108,
'cat_20_pivothead':109,
'cat_21_pivothead':110,
'cat_22_pivothead':111,
'cat_23_pivothead':112,
'cat_24_pivothead':113,
'cat_25_pivothead':114,
'cat_26_pivothead':115,
'cat_27_pivothead':116,
'cat_28_pivothead':117,
'cat_29_pivothead':118,
'cat_30_pivothead':119,
'cup_01_pivothead':120,
'cup_02_pivothead':121,
'cup_03_pivothead':122,
'cup_04_pivothead':123,
'cup_05_pivothead':124,
'cup_06_pivothead':125,
'cup_07_pivothead':126,
'cup_08_pivothead':127,
'cup_09_pivothead':128,
'cup_10_pivothead':129,
'cup_11_pivothead':130,
'cup_12_pivothead':131,
'cup_13_pivothead':132,
'cup_14_pivothead':133,
'cup_15_pivothead':134,
'cup_16_pivothead':135,
'cup_17_pivothead':136,
'cup_18_pivothead':137,
'cup_19_pivothead':138,
'cup_20_pivothead':139,
'cup_21_pivothead':140,
'cup_22_pivothead':141,
'cup_23_pivothead':142,
'cup_24_pivothead':143,
'cup_25_pivothead':144,
'cup_26_pivothead':145,
'cup_27_pivothead':146,
'cup_28_pivothead':147,
'cup_29_pivothead':148,
'cup_30_pivothead':149,
'duck_01_pivothead':150,
'duck_02_pivothead':151,
'duck_03_pivothead':152,
'duck_04_pivothead':153,
'duck_05_pivothead':154,
'duck_06_pivothead':155,
'duck_07_pivothead':156,
'duck_08_pivothead':157,
'duck_09_pivothead':158,
'duck_10_pivothead':159,
'duck_11_pivothead':160,
'duck_12_pivothead':161,
'duck_13_pivothead':162,
'duck_14_pivothead':163,
'duck_15_pivothead':164,
'duck_16_pivothead':165,
'duck_17_pivothead':166,
'duck_18_pivothead':167,
'duck_19_pivothead':168,
'duck_20_pivothead':169,
'duck_21_pivothead':170,
'duck_22_pivothead':171,
'duck_23_pivothead':172,
'duck_24_pivothead':173,
'duck_25_pivothead':174,
'duck_26_pivothead':175,
'duck_27_pivothead':176,
'duck_28_pivothead':177,
'duck_29_pivothead':178,
'duck_30_pivothead':179,
'giraffe_01_pivothead':180,
'giraffe_02_pivothead':181,
'giraffe_03_pivothead':182,
'giraffe_04_pivothead':183,
'giraffe_05_pivothead':184,
'giraffe_06_pivothead':185,
'giraffe_07_pivothead':186,
'giraffe_08_pivothead':187,
'giraffe_09_pivothead':188,
'giraffe_10_pivothead':189,
'giraffe_11_pivothead':190,
'giraffe_12_pivothead':191,
'giraffe_13_pivothead':192,
'giraffe_14_pivothead':193,
'giraffe_15_pivothead':194,
'giraffe_16_pivothead':195,
'giraffe_17_pivothead':196,
'giraffe_18_pivothead':197,
'giraffe_19_pivothead':198,
'giraffe_20_pivothead':199,
'giraffe_21_pivothead':200,
'giraffe_22_pivothead':201,
'giraffe_23_pivothead':202,
'giraffe_24_pivothead':203,
'giraffe_25_pivothead':204,
'giraffe_26_pivothead':205,
'giraffe_27_pivothead':206,
'giraffe_28_pivothead':207,
'giraffe_29_pivothead':208,
'giraffe_30_pivothead':209,
'helicopter_01_pivothead':210,
'helicopter_02_pivothead':211,
'helicopter_03_pivothead':212,
'helicopter_04_pivothead':213,
'helicopter_05_pivothead':214,
'helicopter_06_pivothead':215,
'helicopter_07_pivothead':216,
'helicopter_08_pivothead':217,
'helicopter_09_pivothead':218,
'helicopter_10_pivothead':219,
'helicopter_11_pivothead':220,
'helicopter_12_pivothead':221,
'helicopter_13_pivothead':222,
'helicopter_14_pivothead':223,
'helicopter_15_pivothead':224,
'helicopter_16_pivothead':225,
'helicopter_17_pivothead':226,
'helicopter_18_pivothead':227,
'helicopter_19_pivothead':228,
'helicopter_20_pivothead':229,
'helicopter_21_pivothead':230,
'helicopter_22_pivothead':231,
'helicopter_23_pivothead':232,
'helicopter_24_pivothead':233,
'helicopter_25_pivothead':234,
'helicopter_26_pivothead':235,
'helicopter_27_pivothead':236,
'helicopter_28_pivothead':237,
'helicopter_29_pivothead':238,
'helicopter_30_pivothead':239,
'horse_01_pivothead':240,
'horse_02_pivothead':241,
'horse_03_pivothead':242,
'horse_04_pivothead':243,
'horse_05_pivothead':244,
'horse_06_pivothead':245,
'horse_07_pivothead':246,
'horse_08_pivothead':247,
'horse_09_pivothead':248,
'horse_10_pivothead':249,
'horse_11_pivothead':250,
'horse_12_pivothead':251,
'horse_13_pivothead':252,
'horse_14_pivothead':253,
'horse_15_pivothead':254,
'horse_16_pivothead':255,
'horse_17_pivothead':256,
'horse_18_pivothead':257,
'horse_19_pivothead':258,
'horse_20_pivothead':259,
'horse_21_pivothead':260,
'horse_22_pivothead':261,
'horse_23_pivothead':262,
'horse_24_pivothead':263,
'horse_25_pivothead':264,
'horse_26_pivothead':265,
'horse_27_pivothead':266,
'horse_28_pivothead':267,
'horse_29_pivothead':268,
'horse_30_pivothead':269,
'mug_01_pivothead':270,
'mug_02_pivothead':271,
'mug_03_pivothead':272,
'mug_04_pivothead':273,
'mug_05_pivothead':274,
'mug_06_pivothead':275,
'mug_07_pivothead':276,
'mug_08_pivothead':277,
'mug_09_pivothead':278,
'mug_10_pivothead':279,
'mug_11_pivothead':280,
'mug_12_pivothead':281,
'mug_13_pivothead':282,
'mug_14_pivothead':283,
'mug_15_pivothead':284,
'mug_16_pivothead':285,
'mug_17_pivothead':286,
'mug_18_pivothead':287,
'mug_19_pivothead':288,
'mug_20_pivothead':289,
'mug_21_pivothead':290,
'mug_22_pivothead':291,
'mug_23_pivothead':292,
'mug_24_pivothead':293,
'mug_25_pivothead':294,
'mug_26_pivothead':295,
'mug_27_pivothead':296,
'mug_28_pivothead':297,
'mug_29_pivothead':298,
'mug_30_pivothead':299,
'spoon_01_pivothead':300,
'spoon_02_pivothead':301,
'spoon_03_pivothead':302,
'spoon_04_pivothead':303,
'spoon_05_pivothead':304,
'spoon_06_pivothead':305,
'spoon_07_pivothead':306,
'spoon_08_pivothead':307,
'spoon_09_pivothead':308,
'spoon_10_pivothead':309,
'spoon_11_pivothead':310,
'spoon_12_pivothead':311,
'spoon_13_pivothead':312,
'spoon_14_pivothead':313,
'spoon_15_pivothead':314,
'spoon_16_pivothead':315,
'spoon_17_pivothead':316,
'spoon_18_pivothead':317,
'spoon_19_pivothead':318,
'spoon_20_pivothead':319,
'spoon_21_pivothead':320,
'spoon_22_pivothead':321,
'spoon_23_pivothead':322,
'spoon_24_pivothead':323,
'spoon_25_pivothead':324,
'spoon_26_pivothead':325,
'spoon_27_pivothead':326,
'spoon_28_pivothead':327,
'spoon_29_pivothead':328,
'spoon_30_pivothead':329,
'truck_01_pivothead':330,
'truck_02_pivothead':331,
'truck_03_pivothead':332,
'truck_04_pivothead':333,
'truck_05_pivothead':334,
'truck_06_pivothead':335,
'truck_07_pivothead':336,
'truck_08_pivothead':337,
'truck_09_pivothead':338,
'truck_10_pivothead':339,
'truck_11_pivothead':340,
'truck_12_pivothead':341,
'truck_13_pivothead':342,
'truck_14_pivothead':343,
'truck_15_pivothead':344,
'truck_16_pivothead':345,
'truck_17_pivothead':346,
'truck_18_pivothead':347,
'truck_19_pivothead':348,
'truck_20_pivothead':349,
'truck_21_pivothead':350,
'truck_22_pivothead':351,
'truck_23_pivothead':352,
'truck_24_pivothead':353,
'truck_25_pivothead':354,
'truck_26_pivothead':355,
'truck_27_pivothead':356,
'truck_28_pivothead':357,
'truck_29_pivothead':358,
'truck_30_pivothead':359
}


def read_labels(name ):
    with open( name , 'rb') as f:
        return pickle.load(f)

labels_train = read_labels("/media/alexa/DATA/Datasets/toybox/labels.pkl")


def ordered_glob(rootdir='.', suffix='',split=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    filenames = []
    filenames_folder = []

    folders = glob.glob(rootdir + "/*")

    for folder in folders:

       # folder_id = os.path.split(folder)[1]

       # if split == "train": 
       #        known_set = known_classes[:]
       # else: 
       #        known_set = known_classes_test[:]

       # if folder_id not in known_set:

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



class cnn_toybox(data.Dataset):

    """tless loader 
    """
   

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(224, 224), augmentations=None, class_id=""):
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

        self.files[split] = ordered_glob(rootdir=self.images_base, suffix='.png', split=split)

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



        obj_class = os.path.split(os.path.split(img_path)[0])[1]

        if "hodgepodge" in img_path:
            lbl = np.array([labels_train[obj_class]]) #int(img_path[-10:-8])
        else:
            lbl = np.array([labels_test[obj_class]])



        img = self.resize_keepRatio(img)
        img = np.array(img, dtype=np.uint8)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, img_path

    def resize_keepRatio(self, img):

        old_size = img.size

        ratio = float(self.img_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (self.img_size[0], self.img_size[1]))
        new_im.paste(img, ((self.img_size[0]-new_size[0])//2,
                    (self.img_size[1]-new_size[1])//2))

        return new_im

    def transform(self, img, lbl):
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

        classes = np.unique(lbl)
        # lbl = lbl.astype(float)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    local_path = '/media/mikelf/media_rob/Datasets/emmi'
    dst = cnn_toybox(local_path, split="train", is_transform=True, augmentations=None, class_id='avery_binder')
    bs = 3
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)
    f, axarr = plt.subplots(bs, 1)

    for i, data in enumerate(trainloader):
        imgs, labels_np ,path = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])

        for j in range(bs):
            axarr[j].imshow(imgs[j])
            print(labels_np)

            #print(filenames)
            
        plt.pause(0.1)
        plt.cla()