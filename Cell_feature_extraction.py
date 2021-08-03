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

nuclei_distance_measures = [150, 300, 450]


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
        name_split = files.split(parser)
        second_split = name_split[1].split('.')
        if second_split[0] not in unique_names:
            unique_names.append(second_split[0])
    return unique_names
    

#%% Computational Loop
import time
unique_scan_names = unique_image_names(img_names)
t0 = time.time()
for unique_scan in unique_scan_names: 
    
    print('Slide Name:' + unique_scan)

    for img in img_names:
        img_name_split = img.split('_')
        img_name_split = img_name_split[1].split('.') 
        if img_name_split[0] == unique_scan:

            excel_names = ['hemo-nucleus', 'eosin-nucleus', 'hemo-cyto', 'eosin-cyto']
            
            # only works for specific QuPath Names
            values = img_name_split[1].split('(')
            nums = values[1]
            nums = nums[1:-1]
            offsets = nums.split(',')
            x_offset = offsets[1].split('=')
            x_offset = int(x_offset[1])
            y_offset = offsets[2].split('=')
            y_offset = int(y_offset[1])
            
            w = offsets[3].split('=')
            w = int(w[1])
            h = offsets[4].split('=')
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
                    temp = pd.read_csv(f'{results_root}/{img}HT_{xl}.xls', sep="\t", header = 0 , index_col = 0)
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
    
    
    
# Start Computation loop

    nuc_loc = slide_data[['hemo-nucleus_X', 'hemo-nucleus_Y']]
    slide_date_no_loc = slide_data.drop(columns = ['hemo-nucleus_X', 'hemo-nucleus_Y', 'hemo-nucleus_XM',
                                                    'hemo-nucleus_YM', 'hemo-nucleus_Name'])
    col_names = slide_date_no_loc.columns
    
    num_total_cells = nuc_loc.shape[0]
    
    clusts_divisor = 20
    clust_sizes = []
    
    clusts = int(num_total_cells/clusts_divisor)
    
    while clusts > 10 :
        clust_sizes.append(clusts)
        clusts_divisor = clusts_divisor * 3
        clusts = int(num_total_cells/clusts_divisor)

    nuc_loc_values = nuc_loc.values   
    standerd_scaler = preprocessing.StandardScaler()
    nuc_loc_values = standerd_scaler.fit_transform(nuc_loc_values)       

    for idx2, n_clust in enumerate(clust_sizes):
        spectral = cluster.SpectralClustering(n_clusters = n_clust, eigen_solver = 'arpack',
                                              affinity = "nearest_neighbors")
        if idx2 == 0:
            spectral_labels =spectral.fit(nuc_loc_values).labels_                                                
            spectral_labels =  pd.DataFrame(spectral_labels.reshape([len(spectral_labels), 1]),
                                                                    columns= [f'spectral_{n_clust}_clusters'])
            print(f'spectral_{n_clust}_clusters')
        else:
            spect_temp = spectral.fit(nuc_loc_values).labels_
            spect_temp = pd.DataFrame(spect_temp.reshape([len(spect_temp),1]),
                                      columns= [f'spectral_{n_clust}_clusters'])
            spectral_labels = pd.concat([spectral_labels, spect_temp], axis = 1)
            print(f'spectral_{n_clust}_clusters')


    for idx2, n_clust in enumerate(clust_sizes):
        gmm = mixture.GaussianMixture(n_components= n_clust , covariance_type='full')
        if idx2 == 0:
            gmm_fit = gmm.fit(nuc_loc_values)
            gmm_labels = gmm_fit.predict(nuc_loc_values)
            gmm_labels = pd.DataFrame(gmm_labels.reshape([len(gmm_labels),1]),
                                      columns = [f'gmm_{n_clust}_clusters'])
            print(f'gmm_{n_clust}_clusters')
        else:
            gmm_temp_fit = gmm.fit(nuc_loc_values)
            gmm_temp_labels = gmm_temp_fit.predict(nuc_loc_values)
            gmm_temp_labels = pd.DataFrame(gmm_temp_labels.reshape([len(gmm_temp_labels),1]), 
                                           columns = [f'gmm_{n_clust}_clusters'])
            gmm_labels = pd.concat([gmm_labels, gmm_temp_labels], axis = 1)
            print(f'gmm_{n_clust}_clusters')


    

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


    #Calculate density degree with above function
    
    nuc_loc_dist_degree = density_degree(dist_matrix)
    nuc_degree = pd.DataFrame(np.sum(nuc_loc_dist_degree, axis = 1), columns = ['Degree'])

    new_data_final.reset_index(drop=True, inplace=True)
    slide_data.reset_index(drop=True, inplace=True)
    
    new_data_final = pd.concat([new_data_final, nuc_degree, spectral_labels, gmm_labels], axis = 1)
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

all_data.to_csv(f'{save_path}/AllData.csv')

t1 = time.time()

total_t = t1 - t0


