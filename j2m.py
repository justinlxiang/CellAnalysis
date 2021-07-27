import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import roifile

from tqdm import tqdm

# Location of the original images to get names
file_names = os.listdir('/Users/juxiang/Documents/Work/CellAnalysis/ROIs/json')
file_names = [k[:-5] for k in file_names]


for file_name in file_names:
    path = '/Users/juxiang/Documents/Work/CellAnalysis/ROIs'
    sh_fname = f'{path}/json/{file_name}.json'

    print(file_name)
    print(sh_fname)

    with open(sh_fname, 'r') as f:
        sh_json = json.load(f)

    counter = 0

    if not os.path.isdir(f'{path}/{file_name}/'):
        os.mkdir(f'{path}/{file_name}/')


    for key, value in tqdm(sh_json['nuc'].items()) :

        if len(value['contour']) > 1:
    
            roi = roifile.ImagejRoi.frompoints(value['contour'])
            

                
            roi.tofile(f'{path}/{file_name}/{file_name}_{counter}.roi')

            counter +=1

    print(roi)
