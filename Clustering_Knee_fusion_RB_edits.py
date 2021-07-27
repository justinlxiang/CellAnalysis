import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

from skimage.feature import hog, local_binary_pattern
from skimage import data, exposure

import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
#import snf
#from snf import compute
#import seaborn as sns

#import scipy.cluster.hierarchy as hc
#import scipy.spatial as sp

from scipy.spatial import distance_matrix

import networkx as nx

from sklearn.neighbors import NearestNeighbors

import sklearn
from tqdm import tqdm

from netneurotools import cluster
from sklearn.linear_model import LinearRegression

from sklearn.metrics.pairwise import euclidean_distances
import scipy.cluster.hierarchy as shc
import scipy.spatial as sp
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import umap
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.utils import resample
from sklearn.metrics.pairwise import euclidean_distances
import scipy.cluster.hierarchy as shc
import scipy.spatial as sp
from sknetwork.clustering import Louvain
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE

#%%

results_root = '/Users/juxiang/Documents/Work/CellAnalysis/XLS/'

#%%

#file_names = os.listdir('/Users/juxiang/Documents/Work/CellAnalysis/Images and json files/')
file_names = os.listdir('/Users/juxiang/Documents/Work/CellAnalysis/oneImage/')
file_names = [k[:-4] for k in file_names]

df_data_list_color_nuc = []
df_data_list_cyto = []
df_data_list_shape_nuc = []
df_data_list_clus_identity = []

df_roi_list = []


nuclei_distance_measures = [75, 150, 300]


#%% Functions

def density_degree(dist_matix):
    
    for row in range(len(dist_matix[:, 0])):
        for col in range(len(dist_matix[0, :])):
        
            if dist_matix[row, col] <= 75:
                dist_matix[row, col] = 100
                continue
            
            if 150 >= dist_matix[row, col] > 75:
                dist_matix[row, col] = 10
                continue
            
            if 300 >= dist_matix[row, col] > 150:
                dist_matix[row, col] = 1
                continue
            
            if dist_matix[row, col] > 300:
                dist_matix[row, col] = 0
                continue
        
    return dist_matix





#%%

for file_name in tqdm(file_names):
    fiji_path = f'{results_root}'
    nuc_data = pd.read_csv(f'{fiji_path}{file_name}HT_nucleus.xls', sep="\t", header = 0 , index_col = 0)
    cyto_data = pd.read_csv(f'{fiji_path}{file_name}HT_eosin.xls', sep="\t", header = 0 , index_col = 0)
    #print(nuc_data.Name.values)
    df_roi_list.append(nuc_data.Name.values.flatten())
    #cyto_data = pd.read_csv(fiji_path+'101.vsi - 40x (1, x=19154, y=6126, w=4510, h=4331)_cytoplasm.csv', header = 0 , index_col = 0)
    #print(df_roi_list[0])
    #print(nuc_data.head())
    #print(cyto_data.head())

    cyto_data = cyto_data[['Mean','StdDev','Mode', '%Area', 'Skew', 'Kurt']]
    cyto_data =cyto_data.fillna(0)
    nuc_data_color = nuc_data[['Mean', 'StdDev',  'Skew', 'Kurt', 'RawIntDen' ]]
    nuc_data_shape = nuc_data[[ 'Feret','FeretAngle', 'MinFeret', 'Solidity', 'AR','Area', 'Perim.']]

    nuc_data_loc = nuc_data[['XM', 'YM']]

    

    
    for idx, nuclei_dist in enumerate(nuclei_distance_measures):
        
        find_close_cells = distance_matrix(nuc_data_loc.values, nuc_data_loc.values)
        
        find_close_cells[find_close_cells <= nuclei_dist] = 1
        find_close_cells[find_close_cells > nuclei_dist] = 0
        
        col_names = [f'nucMean_{nuclei_dist}', f'nucStdDev_{nuclei_dist}',  f'nucSkew_{nuclei_dist}',
                     f'nucKurt_{nuclei_dist}', f'nucRawIntDen_{nuclei_dist}',
                     f'cytoMean_{nuclei_dist}', f'cytoStdDev_{nuclei_dist}', f'cytoMode_{nuclei_dist}', 
                     f'cyto%Area_{nuclei_dist}', f'cytoSkew_{nuclei_dist}', f'cytoKurt_{nuclei_dist}',
                     f'Feret_{nuclei_dist}',f'FeretAngle_{nuclei_dist}', f'MinFeret_{nuclei_dist}', f'Solidity_{nuclei_dist}',
                     f'AR_{nuclei_dist}',f'Area_{nuclei_dist}', f'Perim._{nuclei_dist}',
                     f'r-Squared_{nuclei_dist}']
        
        
        cluster_identity_per_dist = pd.DataFrame(columns = col_names)
        close_cells = []
        
        for i in range(len(find_close_cells[:, 0])):
            current_close_cells = np.where(find_close_cells[i, :] == 1)
            close_cells = []
            
            for h in range(len(current_close_cells[0])):
                
                a = nuc_data_loc.loc[current_close_cells[0][h]+1].values
                
                close_cells.append(a)
        
            close_cells = np.array(close_cells)
            r_squared_current_close_cells = 0
            
            if len(close_cells)>5:
                x, y = close_cells[:, 0], close_cells[:, 1]
                
                # Reshaping
                x, y = x.reshape(-1,1), y.reshape(-1, 1)
                
                # Linear Regression Object and Fitting linear model to the data
                lin_regression = LinearRegression().fit(x,y)
                
                r_squared_current_close_cells = lin_regression.score(x,y)
            
            nuc_color_avg = np.zeros(len(nuc_data_color.columns))
            cyto_color_avg = np.zeros(len(cyto_data.columns))
            nuc_shape_avg = np.zeros(len(nuc_data_shape.columns))
            
            for cells in current_close_cells[0]:
                nuc_color_avg = nuc_color_avg + nuc_data_color.iloc[[cells]].values
                cyto_color_avg = cyto_color_avg + cyto_data.iloc[[cells]].values
                nuc_shape_avg = nuc_shape_avg + nuc_data_shape.iloc[[cells]].values
            
            nuc_color_avg = nuc_color_avg/len(current_close_cells[0])
            cyto_color_avg = cyto_color_avg/len(current_close_cells[0])
            nuc_shape_avg = nuc_shape_avg/len(current_close_cells[0])

            sd_nuc_color = np.zeros(len(nuc_data_color.columns))
            sd_cyto_color = np.zeros(len(cyto_data.columns))
            sd_nuc_shape = np.zeros(len(nuc_data_shape.columns))


            for cells in current_close_cells[0]:
                sd_nuc_color = sd_nuc_color + (nuc_data_color.iloc[[cells]].values - nuc_color_avg)**2
                sd_cyto_color = sd_cyto_color + (cyto_data.iloc[[cells]].values - cyto_color_avg)**2
                sd_nuc_shape = sd_nuc_shape + (nuc_data_shape.iloc[[cells]].values - nuc_shape_avg)**2
            
            
            sd_nuc_color = np.sqrt(sd_nuc_color/len(current_close_cells[0]))
            sd_cyto_color = np.sqrt(sd_cyto_color/len(current_close_cells[0]))
            sd_nuc_shape = np.sqrt(sd_nuc_shape/len(current_close_cells[0]))

            for j in range (0,len(sd_nuc_color[0])):
                if sd_nuc_color[0][j] ==0:
                    sd_nuc_color[0][j] = 1

            for j in range (0,len(sd_cyto_color[0])):
                if sd_cyto_color[0][j] ==0:
                    sd_cyto_color[0][j] = 1

            for j in range (0,len(sd_nuc_shape[0])):
                if sd_nuc_shape[0][j] ==0:
                    sd_nuc_shape[0][j] = 1

            nuc_color_Zscore = np.zeros(len(nuc_data_color.columns))
            cyto_color_Zscore = np.zeros(len(cyto_data.columns))
            nuc_shape_Zscore = np.zeros(len(nuc_data_shape.columns))

            nuc_color_Zscore = np.divide((nuc_data_color.iloc[[i]].values  -  nuc_color_avg),sd_nuc_color)  
            cyto_color_Zscore = np.divide((cyto_data.iloc[[i]].values  -  cyto_color_avg),sd_cyto_color)
            nuc_shape_Zscore = np.divide((nuc_data_shape.iloc[[i]].values  -  nuc_shape_avg),sd_nuc_shape)
            
            ## Put shape values into data frame
            clust_data = np.concatenate((nuc_color_avg[0], cyto_color_avg[0], nuc_shape_avg[0], sd_nuc_color[0], sd_cyto_color[0], sd_nuc_shape[0], nuc_color_Zscore[0], cyto_color_Zscore[0], nuc_shape_Zscore[0]))
            clust_data = np.append(clust_data, r_squared_current_close_cells)
    
            
            
            clust_ident_temp = pd.DataFrame(clust_data.reshape((1,-1)), columns = col_names)
            
            cluster_identity_per_dist = cluster_identity_per_dist.append(clust_ident_temp)
    
        if idx == 0:
            cluster_identity = cluster_identity_per_dist
        else:
            cluster_identity = pd.concat([cluster_identity, cluster_identity_per_dist] , axis = 1)
    
    #adfs
        
        
        
        
        
    nuc_data_loc_dist = distance_matrix(nuc_data_loc.values, nuc_data_loc.values)
    nuc_data_loc_dist_degree = density_degree(nuc_data_loc_dist)
    

    
    #print(nuc_data_loc_dist)
    nuc_degree = pd.DataFrame(np.sum(nuc_data_loc_dist_degree,axis = 1), columns = ['Degree'])
    cluster_identity.reset_index(drop=True, inplace=True)
    cluster_identity = pd.concat([cluster_identity, nuc_degree], axis = 1)
    #print(nuc_degree)


    shape_data_df_single = pd.DataFrame(nuc_data_shape, columns = nuc_data_shape.columns)
    color_data_df_single = pd.DataFrame(nuc_data_color, columns = nuc_data_color.columns)

    #print(shape_data_df_single.head())
    df_data_list_color_nuc.append(color_data_df_single)
    df_data_list_cyto.append(cyto_data)
    df_data_list_shape_nuc.append(shape_data_df_single)
    df_data_list_clus_identity.append(cluster_identity)
   
shape_data_nuc_df = pd.concat(df_data_list_shape_nuc)
color_data_nuc_df = pd.concat(df_data_list_color_nuc)
data_cyto_df = pd.concat(df_data_list_cyto)
data_clus_identity_df = pd.concat(df_data_list_clus_identity)

#print(shape_data_nuc_df.head())
#print(color_data_nuc_df.head())
#print(data_cyto_df.head())
#print(data_clus_identity_df.head())


shape_data_nuc_df.reset_index(drop=True, inplace=True)
color_data_nuc_df.reset_index(drop=True, inplace=True)
data_cyto_df.reset_index(drop=True, inplace=True)
data_clus_identity_df.reset_index(drop=True, inplace=True)


all_data = pd.concat([shape_data_nuc_df, color_data_nuc_df, data_cyto_df, data_clus_identity_df], axis = 1)

scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
scaler4 = StandardScaler()

shape_data_nuc_df_scale = scaler1.fit_transform(shape_data_nuc_df.values)
color_data_nuc_df_scale = scaler2.fit_transform(color_data_nuc_df.values)
data_cyto_df_scale = scaler3.fit_transform(data_cyto_df.values)
data__clus_identity_df_scale = scaler4.fit_transform(data_clus_identity_df.values)


data_list = [data_cyto_df_scale, shape_data_nuc_df_scale, color_data_nuc_df_scale, data__clus_identity_df_scale ]#

'''
shape_data_nuc_dis = distance_matrix(shape_data_nuc_df_scale, shape_data_nuc_df_scale)
color_data_nuc_dis = distance_matrix(color_data_nuc_df_scale, color_data_nuc_df_scale)
data_eos_dis = distance_matrix(data_eos_df_scale, data_eos_df_scale)
data_degree_dis = distance_matrix(data_degree_df_scale, data_degree_df_scale)

shape_data_nuc_dis = shape_data_nuc_dis/shape_data_nuc_dis.max()
color_data_nuc_dis = color_data_nuc_dis/color_data_nuc_dis.max()
data_eos_dis = data_eos_dis/data_eos_dis.max()
data_degree_dis = data_degree_dis/data_degree_dis.max()


fin_dis = (0.4*shape_data_nuc_dis)+(0.4*color_data_nuc_dis) + (0.1*data_degree_dis) + (0.1*data_eos_dis)
'''

#from snf import compute
#print('1')
#affinities = compute.make_affinity([shape_data_nuc_df_scale,color_data_nuc_df_scale, data_eos_df_scale, data_degree_df_scale], metric='euclidean')
#print('2')
#fused = compute.snf(affinities)
#print('3')
#best, second = snf.get_n_clusters(fused)
#print(best,second)

#print(shape_data_df_scale)
#shape_data_knn = distance_matrix(shape_data_df_scale, shape_data_df_scale)

#print(shape_data_knn.max())

#shape_data_knn = shape_data_knn/shape_data_knn.max()

#shape_data_knn = sklearn.neighbors.kneighbors_graph(shape_data_df.values, 10, mode='distance', include_self=True)
#shape_data_knn = shape_data_knn.toarray()
#print(shape_data_knn)
#print(dis_mat2)

#dis_mat2 = dis_mat2/dis_mat2.max()
#plt.matshow(dis_mat2)
#plt.show()

#linkage3= hc.linkage(sp.distance.squareform(shape_data_knn), method='ward')
#dend = hc.dendrogram(linkage3)# , labels = shape_data_df.index.values)
#plt.show()
#plt.savefig(f'C:\\Users\\Matt\\Desktop\\Dendrogram_ward.jpg')
#plt.close()

#g = sns.clustermap(shape_data_knn, row_linkage=linkage3, col_linkage=linkage3)
#plt.savefig(f'C:\\Users\\Matt\\Desktop\\Clustergram_ward.jpg')
#plt.close()
#plt.show()


#%% HC Clustering Together 1




#%% HC Clustering Together 2

X = all_data

scaler_standard = StandardScaler()
scaler_standard.fit(X)    
X_scaled = scaler_standard.transform(X)

#remove low variance features
selector = VarianceThreshold()
X_scaled = selector.fit_transform(X_scaled)

print("Debug")

# a. PCA
pca = PCA(random_state=1)

# fit the PCA model using training set
pca.fit(X_scaled)

X_scaled_pca = pca.transform(X_scaled)

print("Debug0")
plt.scatter(X_scaled_pca[:,0], X_scaled_pca[:,1])
plt.show(block=False)
print("Debug1")


# Plot Percent Explained
plt.plot(np.cumsum(pca.explained_variance_ratio_))
print("Debug2")
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show(block=False)
print("Debug3")


# selected components selection
pc_cum_sum = np.cumsum(pca.explained_variance_ratio_)

for i in range(len(pc_cum_sum)):
    if pc_cum_sum[i] > 0.9:
        pcs_picked = i
        break

print(pcs_picked)

X_pcs_picked =  X_scaled_pca[:, :pcs_picked]

#%% UMAP
n_neighbors_calc = int((pcs_picked/2) + 1)

X_display_UMAP = umap.UMAP(n_neighbors= n_neighbors_calc,
                       min_dist=.000001,
                       n_components = 2,
                       metric='euclidean').fit_transform(X_pcs_picked)

plt.scatter(X_display_UMAP[:,0], X_display_UMAP[:,1],
            s = 0.02)
plt.show(block=False)
print("Debug4")



#%% HC Clustering Together 3

dist_mat = euclidean_distances(X_pcs_picked)

# #im not sure why i am not getting a symetric matrix but i was and the linkage calculation was not working
# # so i rounded the data set to 5 decimal places as i noticed the asymetry was vanishingly small
dist_mat  = np.round(dist_mat, 4)

#dist_mat.shape


# Generate dendrogram to determine the optimal cluster number

linkage = shc.linkage(sp.distance.squareform(dist_mat), method='ward')
 # options for method: average, single complete, ward, etc. 
                     

#linkage = shc.linkage(dist_mat, method='ward') # options for method: average, single,
                                     #                     complete, ward, etc. 
                     

den_plt = shc.dendrogram(linkage, orientation='top', truncate_mode='level', p = 8)

plt.show(block=False)
#%%#%% Cut dendrogram

labels_hc_Low = shc.fcluster(Z = linkage, # input linkage
                         t = 5, # cluster number
                         criterion = 'maxclust'
                        )

labels_hc_Mid = shc.fcluster(Z = linkage, # input linkage
                         t = 9, # cluster number
                         criterion = 'maxclust'
                        )


labels_hc_High = shc.fcluster(Z = linkage, # input linkage
                         t = 15, # cluster number
                         criterion = 'maxclust'
                        )

#%% Plot with HC clusters

unique_labels, lanel_cnts = np.unique(labels_hc_High, return_counts=True)
for l in unique_labels:  
    plt.scatter(X_display_UMAP[labels_hc_High == l, 0], X_display_UMAP[labels_hc_High == l, 1], 
                s=.5, # marker size
                alpha=0.5, # transparency
                label='Cluster %s' % l, # label
                )
plt.legend(bbox_to_anchor=(1,1), loc="upper left", markerscale = 20)
plt.xlabel('UMAP_0')
plt.ylabel('UMAP_1')
plt.show(block=False)

#%% HC Clustering Together 4

cluster_name = 'Together_Low'

rois = np.concatenate(df_roi_list,axis=0)

d = {'Name':rois,'Cluster_Label':labels_hc_Low}

df = pd.DataFrame(d)
df.to_csv(f'{results_root}/Result_clustering_{cluster_name}.csv')


#######

cluster_name = 'Together_Mid'

d = {'Name':rois,'Cluster_Label':labels_hc_Mid}

df = pd.DataFrame(d)
df.to_csv(f'{results_root}/Result_clustering_{cluster_name}.csv')


############

cluster_name = 'Together_High'

d = {'Name':rois,'Cluster_Label':labels_hc_High}

df = pd.DataFrame(d)
df.to_csv(f'{results_root}/Result_clustering_{cluster_name}.csv')


#%% Consensus Clustering
rois = np.concatenate(df_roi_list,axis=0)
print(rois.shape)

from sklearn.cluster import AgglomerativeClustering
from functools import reduce

# clusters = [2,3,4,5,6,9]
# clust_names = ['coarsest', 'coarser', 'coarse', 'fine', 'finer','finest']

clusters = [5, 9, 15]
clust_names = ['coarsest', 'coarse','finest']

for cq, q in enumerate(clusters):
    cluster_labels = np.zeros((data__clus_identity_df_scale.shape[0],len(data_list)))
    for enum, xyz in enumerate(data_list):
        clustering = AgglomerativeClustering(n_clusters = q, affinity = 'euclidean', linkage = 'ward').fit(xyz)
        cluster_labels[:,enum] = clustering.labels_
    #print(names)
    #print(clustering.labels_)
    print('Consensus...')
    consensus_clustering_labels =  cluster.find_consensus(cluster_labels, seed = 1234)

    d = {'Name':rois,'Cluster_Label':consensus_clustering_labels}

    df = pd.DataFrame(d)
    df.to_csv(f'{results_root}/Result_clustering_{clust_names[cq]}.csv')


        #sim_mat1 = sim_mat1[:1000,:1000]


    #sim_mat2 = 1-dis_mat2


    #sim_mat2 = sim_mat2[:1000,:1000]
    #sim_mats = [sim_mat1,sim_mat2]




    #fused = compute.snf(sim_mats)

'''
    g2 = nx.from_numpy_matrix(np.array(dis_mat_knn) )
    color_map = []

    for node in g2: 
        color_map.append('blue')

    fig, ax = plt.subplots(1, 1, figsize=(40,40))

    nx.draw(g2, node_color=color_map, node_size = 1)

    plt.show()
'''
    
