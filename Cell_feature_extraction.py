#%% Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from skimage.feature import hog, local_binary_pattern
from skimage import data, exposure
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import math
#import snf
#from snf import compute
#import seaborn as sns
#import scipy.cluster.hierarchy as hc
#import scipy.spatial as sp
from scipy import stats
from scipy.spatial import distance_matrix
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import sklearn
from tqdm import tqdm
from netneurotools import cluster
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn import cluster, mixture, preprocessing
from PIL import Image
from libtiff import TIFF
from skimage import io
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

#%% Paths
results_root = '/Users/juxiang/Documents/Work/CellAnalysis/XLS/'
images_path = '/Users/juxiang/Documents/Work/CellAnalysis/Images/'
save_path = '/Users/juxiang/Documents/Work/CellAnalysis/Clusters/'

#%% Hyperparameters and Empty Lists
img_names = os.listdir(images_path)
img_names = [k[:-4] for k in img_names]
scan_data = []
df_roi_list = []
maskNum = np.empty((0,1))

nuclei_distance_measures = [150, 300]# 450]


#%% Functions
def density_degree(dist_matix):
    for row in range(len(dist_matix[:, 0])):
        for col in range(len(dist_matix[0, :])):
            if dist_matix[row, col] <= 150:
                dist_matix[row, col] = 100
                continue
            
            if 300 >= dist_matix[row, col] > 150:
                dist_matix[row, col] = 10
                continue
            
            if 450 >= dist_matix[row, col] > 300:
                dist_matix[row, col] = 1
                continue
            
            if dist_matix[row, col] > 450:
                dist_matix[row, col] = 0
                continue
    return dist_matix



def unique_image_names(file_names_list, parser = '_', parse_index = 1):
    unique_names = []
    for files in file_names_list:
        unique_names.append(files)
        '''name_split = files.split(parser)
        second_split = name_split[1].split('.')
        if second_split[0] not in unique_names:
            unique_names.append(second_split[0])'''
    return unique_names
    

#%% Computational Loop
import time
unique_scan_names = unique_image_names(img_names)
t0 = time.time()
for unique_scan in unique_scan_names: 
    
    print('Slide Name:' + unique_scan)

    for img in img_names:
        if img == unique_scan:

            excel_names = ['hemo-nucleus', 'eosin-nucleus', 'hemo-cyto', 'eosin-cyto']
            
            # only works for specific QuPath Names
            values = img.split('[')
            nums = values[1]
            nums = nums[0:-1]
            offsets = nums.split(',')
            x_offset = offsets[0].split('=')
            x_offset = int(x_offset[1])
            y_offset = offsets[1].split('=')
            y_offset = int(y_offset[1])
            
            w = offsets[2].split('=')
            w = int(w[1])
            h = offsets[3].split('=')
            h = int(h[1])
            img_Y = h
            img_X = w
            
            # excel import loop
            for xl in excel_names:
                if xl == excel_names[0]:
                    img_data = pd.read_csv(f'{results_root}/{img}_{xl}.xls', sep="\t", header = 0 , index_col = 0)
                    img_data = img_data.drop(columns = ['AR', 'FeretX','FeretY', 'IntDen','RawIntDen',
                                                        '%Area', 'MinThr', 'MaxThr'])

                    x = img_data['X'].values
                    y = img_data['Y'].values
                    
                    # Add offset to x and y coodinates
                    
                    # not sure which way is faster
                    #x = np.add(x, x_offset).tolist()
                    #y = np.add(y, y_offset).tolist()
                    
                    # or this
                    x = [f + x_offset for f in x]
                    y = [f + y_offset for f in y]
                    
                    
                    img_data.loc[:, 'X'] = x
                    img_data.loc[:, 'Y'] = y                    
                    
                    #img_data = img_data[['X', 'Y', 'Mean']]
                    
                    img_data = img_data.add_prefix(f'{xl}_')
                    
                else:
                    temp = pd.read_csv(f'{results_root}/{img}_{xl}.xls', sep="\t", header = 0 , index_col = 0)
                    # temp = temp.drop(columns = ['Area', 'Mean', 'StdDev',
                    #                             'Mode', 'Min', 'Max',
                    #                             'X', 'Y', 'XM', 'YM',
                    #                             'Perim.', 'Circ.', 'Feret',                                               
                    #                             'IntDen', 'Median', 'Skew',
                    #                             'Kurt', '%Area', 'RawIntDen',
                    #                             'FeretX', 'FeretY',
                    #                             'MinFeret', 'AR',
                    #                             'FeretAngle', 'Round', 'Solidity',
                    #                              'MinThr', 'MaxThr', 'Name'])
                    temp = temp.drop(columns = [
                                                'X', 'Y', 'XM', 'YM',
                                                'Perim.', 'Circ.', 'Feret',                                               
                                                'IntDen', 
                                                 '%Area', 'RawIntDen',
                                                'FeretX', 'FeretY',
                                                'MinFeret', 'AR',
                                                 'Round', 'Solidity',
                                                  'MinThr', 'MaxThr', 'Name'])                   
                    #temp = temp[['Mean']]
                    temp = temp.add_prefix(f'{xl}_')
                    img_data = pd.concat([img_data, temp], axis = 1)
                    del(temp)
                
                
                    
            # remove cells at the edge of images
            img_nuc_loc = img_data[['hemo-nucleus_X', 'hemo-nucleus_Y']].values
            
            cells_to_remove = []
            
            for idx, cell_loc in enumerate(img_nuc_loc):
            
                if x_offset < cell_loc[0] <  x_offset + 40:
                    cells_to_remove.append(idx + 1)
                    continue
                
                if (img_X - 40) + x_offset < cell_loc[0] <  img_X + x_offset:
                    cells_to_remove.append(idx + 1)
                    continue
                
                if y_offset < cell_loc[1] <  y_offset + 40:
                    cells_to_remove.append(idx + 1)
                    continue
                    
                if (img_Y - 40 ) + y_offset < cell_loc[1] < img_Y + y_offset:
                    cells_to_remove.append(idx +1)
                    continue
                    
            img_data = img_data.drop(index = cells_to_remove)
                

            
            if 'slide_data' not in locals():
                slide_data = img_data
                
            else:
                slide_data = pd.concat([slide_data, img_data], axis = 0)
    
    
    # potential to downsample here if computation is to much
    #slide_data = resample(slide_data, n_samples = 10000)       
    
    
    
#%% Start Computation loop

    nuc_loc = slide_data[['hemo-nucleus_X', 'hemo-nucleus_Y']]
    slide_date_no_loc = slide_data.drop(columns = ['hemo-nucleus_X', 'hemo-nucleus_Y', 'hemo-nucleus_XM',
                                                    'hemo-nucleus_YM', 'hemo-nucleus_Name'])
    col_names = slide_date_no_loc.columns
    

    tree = KDTree(img_nuc_loc)


    find_close_cells_parent = distance_matrix(nuc_loc.values, nuc_loc.values)
    
    dist_matrix = find_close_cells_parent.copy()
    
    find_close_cells = find_close_cells_parent.copy()


    #For distance hyper paramerter    
    for idx, nuclei_dist in enumerate(nuclei_distance_measures):
        
        print('Current Distance Measurment:')
        print(nuclei_dist)
        
        find_close_cells = find_close_cells_parent.copy()
        
        find_close_cells[find_close_cells <= nuclei_dist] = 1
        find_close_cells[find_close_cells > nuclei_dist] = 0      
        
        for i in tqdm(range(len(find_close_cells[:, 0]))):
            current_close_cells = np.where(find_close_cells[i, :] == 1)
            close_cells = []
            
            for h in range(len(current_close_cells[0])):
               
                close_cells.append(nuc_loc.iloc[current_close_cells[0][h]].values)
        
            close_cells = np.array(close_cells)
            r_squared_current_close_cells = np.zeros([1,1])
            
            if len(close_cells)>5:
                x, y = close_cells[:, 0], close_cells[:, 1]
                
                # Reshaping
                x, y = x.reshape(-1,1), y.reshape(-1, 1)
                
                # Linear Regression Object and Fitting linear model to the data
                lin_regression = LinearRegression().fit(x,y)
                
                r_squared_current_close_cells = np.array(lin_regression.score(x,y))
                
                # r_squared_current_close_cells = pd.DataFrame(r_squared_current_close_cells.reshape(1,1),
                #                                              columns = [f'{nuclei_dist}_r_squared'])
            

            
            for cells in current_close_cells[0]:
                
                if cells == current_close_cells[0][0]:
                    close_cell_data = slide_date_no_loc.iloc[[cells]].values
                else:    
                    close_cell_data = np.append(close_cell_data, slide_date_no_loc.iloc[[cells]].values, axis = 0)
            
                
            # Calculate interesting features
            avg = close_cell_data.mean(axis = 0)
            sd = close_cell_data.std(axis = 0)

            Zscore = np.divide((slide_date_no_loc.iloc[[i]].values - avg),sd)
            
            #Zscore[Zscore == 1] = 2
            Zscore = np.nan_to_num(Zscore, nan=0.0)
            # for j in range (0,len(sd)):
            #     if sd[j] ==0:
            #         Zscore[0][j] = 0
            #         print(Zscore[0][j])

            ratio_change = np.divide((slide_date_no_loc.iloc[[i]].values - avg), avg)

            # From Scipy Packag https://docs.scipy.org/doc/scipy/reference/stats.html?highlight=stats#module-scipy.stats
            skew = stats.skew(close_cell_data, axis = 0)
            moderesult = stats.mode(close_cell_data, axis = 0)
            mode = moderesult[0]
            iqr = stats.iqr(close_cell_data, axis = 0)
            kurtosis = stats.kurtosis(close_cell_data, axis = 0)
            sem = stats.sem(close_cell_data, axis = 0)
            entropy = stats.entropy(abs(close_cell_data), axis = 0)

            #Computing Distance of String of Closests Cells
            visited = dict([(i,i)])
            
            dist, ix = tree.query([img_nuc_loc[i][0], img_nuc_loc[i][1]], k=[2], distance_upper_bound = nuclei_dist)
            points = np.empty((0,2))
            points = np.append(points, [[img_nuc_loc[i][0], img_nuc_loc[i][1]]], axis = 0)
            total_distance = 0
            
            if(dist != float("inf")):
                total_distance = dist[0]
                row = ix[0]
                points = np.append(points, [[img_nuc_loc[row][0], img_nuc_loc[row][1]]], axis=0)
                visited[row] = row
                
                for j in range(0,30):
                    #print(dist, ix)
                    count = 0
                    while(ix[0] in visited):
                        dist, ix = tree.query([img_nuc_loc[row][0], img_nuc_loc[row][1]], k=[count + 2], distance_upper_bound = nuclei_dist)
                        count += 1
                        if(dist == float("inf")):
                            break
                    if(dist == float("inf")):
                            break
                    total_distance += dist[0]                                   
                    row = ix[0]
                    points = np.append(points, [[img_nuc_loc[row][0], img_nuc_loc[row][1]]], axis=0)
                    visited[row] = row


            #Area of Convex Hull
            if(points.shape[0] > 2):        
                polygon = ConvexHull(points)
                area = polygon.area
            else:
                area = 0
            
            
            r_squared_string_cells = np.zeros([1,1])
            
            if len(points)>6:
                x, y = points[:, 0], points[:, 1]
                
                # Reshaping
                x, y = x.reshape(-1,1), y.reshape(-1, 1)
                
                # Linear Regression Object and Fitting linear model to the data
                lin_regression = LinearRegression().fit(x,y)
                
                r_squared_string_cells = np.array(lin_regression.score(x,y))



            total_distance = np.array(total_distance)
            area = np.array(area)


            # Add all new calculations to this list
            data_to_join = [avg, sd, Zscore, iqr, sem, ratio_change, entropy]
            data_to_join_names = ['avg', 'sd', 'Zscore', 'iqr', 'sem', 'ratio_change', 'entropy'] 
            
            for idx_2, joining_data in enumerate(data_to_join):
                if idx_2 == 0:
                    new_data_temp1 = pd.DataFrame(joining_data.reshape((1,-1)), columns = col_names)
                    #name_prefix = data_to_join_names[idx_2]
                    new_data_temp1 = new_data_temp1.add_prefix(f'{nuclei_dist}_{data_to_join_names[idx_2]}_')
                else:
                    temp = pd.DataFrame(joining_data.reshape((1,-1)), columns = col_names)
                    temp = temp.add_prefix(f'{nuclei_dist}_{data_to_join_names[idx_2]}_')
                    new_data_temp1 = pd.concat([new_data_temp1, temp], axis = 1)
 
    

            new_data_temp1 = pd.concat([new_data_temp1, pd.DataFrame(r_squared_current_close_cells.reshape(1,1),
                                                                      columns = [f'{nuclei_dist}_r_squared_close']
                                                                      ), pd.DataFrame(r_squared_string_cells.reshape(1,1),
                                                                      columns = [f'{nuclei_dist}_r_squared_nearest']),
                                                                      pd.DataFrame(total_distance.reshape(1,1), columns = [f'{nuclei_dist}_spatialDistance']),
                                                                      pd.DataFrame(area.reshape(1,1), columns = [f'{nuclei_dist}_spatialArea'])], axis = 1)
            
    
            if i == 0:
                new_data = new_data_temp1
            else: 
                new_data = pd.concat([new_data, new_data_temp1], axis = 0)
        
        
        if idx == 0:
            new_data_final = new_data
        else: 
            new_data_final = pd.concat([new_data_final, new_data], axis = 1)            


#%%
    file = f'{unique_scan}-labelled.tif'
    im = io.imread(f'/Users/juxiang/Documents/Work/CellAnalysis/Masks/{file}')
    nans =0
    
    nuc_loc = slide_data[['hemo-nucleus_X', 'hemo-nucleus_Y']]
    for index, row in nuc_loc.iterrows():
        xcor = row['hemo-nucleus_X']
        ycor = row['hemo-nucleus_Y']
        found = False
        for mask in range(12):
             if im[mask][round(ycor)-y_offset][round(xcor)-x_offset] == 255:
                 maskNum = np.append(maskNum, [[mask]], axis=0)
                 found = True
                 break
        if not found:
            maskNum = np.append(maskNum, [[0]], axis=0)
            nans += 1
    

    
#%%   

    #Calculate density degree with above function
    
    nuc_loc_dist_degree = density_degree(dist_matrix)
    nuc_degree = pd.DataFrame(np.sum(nuc_loc_dist_degree, axis = 1), columns = ['Degree'])
    maskNum = pd.DataFrame(maskNum, columns = ['TissueMask'])
    
    new_data_final.reset_index(drop=True, inplace=True)
    slide_data.reset_index(drop=True, inplace=True)
    
    new_data_final = pd.concat([new_data_final, nuc_degree, maskNum], axis = 1)
    slide_data = pd.concat([slide_data, new_data_final], axis = 1)
    
    if unique_scan == unique_scan_names[0]:
        all_data = slide_data
        del(slide_data)
    else:
        all_data = pd.concat([all_data, slide_data], axis = 0)
        del(slide_data)
    #all_data = slide_data
    #break
    del(dist_matrix, find_close_cells_parent)
    
    
save_name = '_'.join(str(e) for e in nuclei_distance_measures)


#%%
data_path = '/Users/juxiang/Documents/Work/CellAnalysis/Data/'

all_data.to_csv(f'{data_path}/AllData.csv')

HeNuc = []
EoNuc = []
HeCyto = []
EoCyto = []
Shape = []

d1HN = []
d1EN = []
d1HC = []
d1EC = []
d1Shape = []
d1Density = []

d2HN = []
d2EN = []
d2HC = []
d2EC = []
d2Shape = []
d2Density = []

for ind, column in enumerate(all_data.columns):
    if "Area" in column or "Perim" in column or "Circ" in column or "Feret" in column or "Solidity" in column or "Round" in column:
        if "150" in column:
            d1Shape.append(ind)
        elif "300" in column:
            d2Shape.append(ind)
        else: 
            Shape.append(ind)


    if "r_squared" in column or "spatial" in column or "Degree" in column:
        if "150" in column:
            d1Density.append(ind)
        elif "300" in column:
            d2Density.append(ind)

    else:
        if "150" in column:
            if "hemo-nucleus" in column:
                d1HN.append(ind)
            elif "eosin-nucleus" in column:
                d1EN.append(ind)
            elif "hemo-cyto" in column:
                d1HC.append(ind)
            elif "eosin-cyto" in column:
                d1EC.append(ind)
        elif "300" in column:
            if "hemo-nucleus" in column:
                d2HN.append(ind)
            elif "eosin-nucleus" in column:
                d2EN.append(ind)
            elif "hemo-cyto" in column:
                d2HC.append(ind)
            elif "eosin-cyto" in column:
                d2EC.append(ind)
        else:
            if "hemo-nucleus" in column:
                HeNuc.append(ind)
            elif "eosin-nucleus" in column:
                EoNuc.append(ind)
            elif "hemo-cyto" in column:
                HeCyto.append(ind)
            elif "eosin-cyto" in column:
                EoCyto.append(ind)

HemoNuc = all_data[all_data.columns[HeNuc]]
EosinHuc = all_data[all_data.columns[EoNuc]]
HemoCyto = all_data[all_data.columns[HeCyto]]
EosinCyto = all_data[all_data.columns[EoCyto]]
Shape =  all_data[all_data.columns[Shape]]

D1HemoNuc = all_data[all_data.columns[d1HN]]
D1EosinNuc = all_data[all_data.columns[d1EN]]
D1HemoCyto = all_data[all_data.columns[d1HC]]
D1EeosinCyto = all_data[all_data.columns[d1EC]]
D1Shape = all_data[all_data.columns[d1Shape]]
D1Density = all_data[all_data.columns[d1Density]]

D2HemoNuc = all_data[all_data.columns[d2HN]]
D2EosinNuc = all_data[all_data.columns[d2EN]]
D2HemoCyto = all_data[all_data.columns[d2HC]]
D2EeosinCyto = all_data[all_data.columns[d2EC]]
D2Shape = all_data[all_data.columns[d2Shape]]
D2Density = all_data[all_data.columns[d2Density]]



HemoNuc.to_csv(f'{data_path}/HemoNuc.csv')
EosinHuc.to_csv(f'{data_path}/EosinHuc.csv')
HemoCyto.to_csv(f'{data_path}/HemoCyto.csv')
EosinCyto.to_csv(f'{data_path}/EosinCyto.csv')
Shape.to_csv(f'{data_path}/Shape.csv')

D1HemoNuc.to_csv(f'{data_path}/D1HemoNuc.csv')
D1EosinNuc.to_csv(f'{data_path}/D1EosinNuc.csv')
D1HemoCyto.to_csv(f'{data_path}/D1HemoCyto.csv')
D1EeosinCyto.to_csv(f'{data_path}/D1EeosinCyto.csv')
D1Shape.to_csv(f'{data_path}/D1Shape.csv')
D1Density.to_csv(f'{data_path}/D1Density.csv')

D2HemoNuc.to_csv(f'{data_path}/D2HemoNuc.csv')
D2EosinNuc.to_csv(f'{data_path}/D2EosinNuc.csv')
D2HemoCyto.to_csv(f'{data_path}/D2HemoCyto.csv')
D2EeosinCyto.to_csv(f'{data_path}/D2EeosinCyto.csv')
D2Shape.to_csv(f'{data_path}/D2Shape.csv')
D2Density.to_csv(f'{data_path}/D2Density.csv')



t1 = time.time()

total_t = t1 - t0




# %%
