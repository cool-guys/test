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

lis = [2236,2244,3213,3244,3248,4206,4243,4244,4248,5210,5211,5212,5213,5219,5225,5226,5229,5235,5243,6237,6243,6244,6247,6248,7242,7244,7212,8210,8212,8235,8243]
for i in lis:
    if i < 1000:
        a = int(i/10)
        b = int(i%10)
    else:
        a = int(i/1000)
        b = int(i%1000)
    os.remove('./DATA/Video/{}/{}train_{}'.format(a,a,b))
    os.remove('./DATA/Video/{}/{}train_{}.mp4'.format(a,a,b))