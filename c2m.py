import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import roifile

from tqdm import tqdm
import read_roi


#%%

image_loc = '/Users/juxiang/Documents/Work/CellAnalysis/Images/'


file_names = os.listdir(image_loc)
file_names = [k[:-4] for k in file_names]



#%%

for file_name in file_names:
    path = '/Users/juxiang/Documents/Work/CellAnalysis/XLS/'
    sh_fname_roi = f'/Users/juxiang/Documents/Work/CellAnalysis/ROIs/{file_name}/'
    roi_files = os.listdir(sh_fname_roi)
    #print(sorted(roi_files))

    roi_files = sorted(roi_files)
    img_fname = f'{image_loc}{file_name}.jpg'

    img = cv2.imread(img_fname,cv2.IMREAD_COLOR)

    clust_names =['High', 'Low', 'Mid', 'Spectral']#, 'coarse', 'fine', 'finer','finest']

    for q in clust_names:
        clust = pd.read_csv(f'/Users/juxiang/Documents/Work/CellAnalysis/Clusters/Result_clustering_{q}.csv', index_col = 0, header = 0)
        clust['NameShort'] = ["_".join(k.split('_')[:2]) for k in clust.Name]
        #print(clust.NameShort.values[0], file_name)
        #print(clust.head())
        #print(clust['NameShort'].str.contains(str(file_name.split('_')[0]), case=False))
        clust = clust.loc[clust['NameShort'].str.contains(str(file_name.split('_')[0]), case=False)]
        #print(clust)
        

        data = pd.read_csv(f'/Users/juxiang/Documents/Work/CellAnalysis/XLS/{file_name}HT_hemo-nucleus.xls', sep="\t", header = 0 , index_col = 0)

        def getList(dict):
            
            return [*dict]

        counter = 0

        if not os.path.isdir(path + file_name):
            os.mkdir(path + file_name)

        color_list =[(255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0), (255,165,0), (128,0,0),
                     (128,0,128), (255,255,255), (153, 76, 0), (0,0,0), (128,128,0), (75,0,130), (244,164,96), (200,100,100) ]

        for count, j in enumerate(roi_files):

            

            roi2 = read_roi.read_roi_file(sh_fname_roi+j)
            #print(sh_fname_roi+j)

            #if getList(roi2)[0] == '4':
            #    print(count)
            #    print(roi2)
            #    break
            

            roi2['coordinates'] = np.column_stack((roi2[getList(roi2)[0]]['x'],roi2[getList(roi2)[0]]['y'] ))

            idx_file = clust.loc[clust['Name'] == j]
            #print(idx_file)
            if(len(idx_file) == 0):
                continue

            #print(color_list[clust.iloc[int(getList(roi2)[0]),1]])
            #print(np.array(np.array(value['contour'])),  color_list[clust.iloc[counter,1]] )
            #roi2 = roifile.ImagejRoi.fromfile(sh_fname_roi+j)
            cv2.drawContours(img, np.array([roi2['coordinates']]), -1, color_list[int(idx_file.Cluster_Labels.values)],3 )
            font = cv2.FONT_HERSHEY_SIMPLEX

            data_file = data.loc[data['Name'] == j]
            #print(data_file)
            #print()

            name_roi = new_string =idx_file.Name.values[0].replace("'", "")
            name_roi = name_roi.split("_")[-1]
            #print(name_roi[:-4])
            #print((int(data_file.XM.values[0]),int(data_file.YM.values[0])))
            #print(idx_file.Category.values[0])
            cv2.putText(img,str(name_roi[:-4]),(int(data_file.XM.values[0]),int(data_file.YM.values[0])), font, 0.3,  color_list[int(idx_file.Cluster_Labels.values)], 1 ,cv2.LINE_AA)




        cv2.imwrite(f'/Users/juxiang/Documents/Work/CellAnalysis/Clusters/{file_name}_{q}_overlay.jpg', img)
