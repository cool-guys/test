import pandas as pd
import numpy as np
import os
import cv2

import random
import Augmentor
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import math
from math import floor, ceil
from scipy.ndimage import gaussian_filter
import time

import glob
'''
lis = []
for i in lis:
    if i < 1000:
        

        a = int(i/100)
        b = int(i%100)
    else:
        a = int(i/1000)
        b = int(i%1000)
    os.remove('./DATA/Video/{}/{}train_{}'.format(a,a,b))
    os.remove('./DATA/Video/{}/{}train_{}.mp4'.format(a,a,b))
'''
f= open('./DATA/image/test/change', 'r')
while True:
    line = f.readline()
    if not line: break
    a = line[0]
    print(a)
    num_list = list(map(int,line[1:].split()))
    print(list(num_list))
    for i in num_list:
        print(i)
        os.remove('../Video/{}/{}train_{}'.format(a,a,i))
        os.remove('../Video/{}/{}train_{}.mp4'.format(a,a,i))
f.close()