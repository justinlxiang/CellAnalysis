#%% Clustering Imports#%%
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
# GIT Hub Commitss
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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from sklearn.metrics.pairwise import euclidean_distances
import scipy.cluster.hierarchy as shc
import scipy.spatial as sp
#from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import umap
#import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#from sklearn.cluster import DBSCAN
#from sklearn.utils import resample
#from sklearn.metrics.pairwise import euclidean_distances
#import scipy.cluster.hierarchy as shc
#import scipy.spatial as sp
#from sknetwork.clustering import Louvain
#from sklearn.cluster import KMeans
#from sklearn import metrics
#import seaborn as sns
from sklearn.impute import SimpleImputer
#from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering


#%% PCA and PCs Picked

results_root = '/Users/juxiang/Documents/Work/CellAnalysis/XLS/'
images_path = '/Users/juxiang/Documents/Work/CellAnalysis/Images/'
save_path = '/Users/juxiang/Documents/Work/CellAnalysis/Clusters/'


all_data = pd.read_csv(f'{save_path}/AllData.csv', index_col = 0)#, sep="\t", header = 0 , index_col = 0)
all_data_trimed = all_data.drop(columns = ['hemo-nucleus_X', 'hemo-nucleus_Y', 'hemo-nucleus_XM', 'hemo-nucleus_YM', 'hemo-nucleus_Name'])

X = all_data_trimed.values

imputer = SimpleImputer(missing_values=np.nan,
                        strategy='constant', fill_value = 0)
imputer.fit(X)

X = imputer.transform(X)

scaler_standard = StandardScaler()
scaler_standard.fit(X)    
X_scaled = scaler_standard.transform(X)

#remove low variance features
selector = VarianceThreshold()
X_scaled = selector.fit_transform(X_scaled)



# a. PCA
pca = PCA(random_state=1)

# fit the PCA model using training set
pca.fit(X_scaled)

X_scaled_pca = pca.transform(X_scaled)

plt.scatter(X_scaled_pca[:,0], X_scaled_pca[:,1])
plt.show()

# Plot Percent Explained
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# selected components selection
pc_cum_sum = np.cumsum(pca.explained_variance_ratio_)

for i in range(len(pc_cum_sum)):
    if pc_cum_sum[i] > 0.8:
        pcs_picked = i
        break


X_pcs_picked =  X_scaled_pca[:, :pcs_picked]

#%% UMAP
n_neighbors_calc = int((pcs_picked/2) + 1)

X_display_UMAP = umap.UMAP(n_neighbors = n_neighbors_calc,
                       min_dist=.0000001,
                       n_components = 2,
                       metric='euclidean').fit_transform(X_pcs_picked)

plt.scatter(X_display_UMAP[:,0], X_display_UMAP[:,1],
            s = 0.05)
plt.show()

#%% Agglomerative Clustering and Dendrogram

dist_mat = euclidean_distances(X_pcs_picked)

# #im not sure why i am not getting a symetric matrix but i was and the linkage calculation was not working
# # so i rounded the data set to 5 decimal places as i noticed the asymetry was vanishingly small
dist_mat  = np.round(dist_mat, 5)

#dist_mat.shape


# Generate dendrogram to determine the optimal cluster number

linkage = shc.linkage(sp.distance.squareform(dist_mat), method='ward')
 # options for method: average, single complete, ward, etc. 
                     

#linkage = shc.linkage(dist_mat, method='ward') # options for method: average, single,
                                     #                     complete, ward, etc. 
                     

den_plt = shc.dendrogram(linkage, orientation='top', truncate_mode='level', p = 8)

plt.show()




#%%#%% Cut dendrogram

spectral = SpectralClustering(n_clusters = 10, affinity = 'nearest_neighbors').fit(X_pcs_picked)
labels_spectral = spectral.labels_


labels_hc_Low = shc.fcluster(Z = linkage, # input linkage
                         t = 6, # cluster number
                         criterion = 'maxclust'
                        )

labels_hc_Mid = shc.fcluster(Z = linkage, # input linkage
                         t = 8, # cluster number
                         criterion = 'maxclust'
                        )


labels_hc_High = shc.fcluster(Z = linkage, # input linkage
                         t = 12, # cluster number
                         criterion = 'maxclust'
                        )


#%% Plot with HC clusters

unique_labels, lanel_cnts = np.unique(labels_hc_Mid, return_counts=True)
for l in unique_labels:  
    plt.scatter(X_display_UMAP[labels_hc_Mid == l, 0], X_display_UMAP[labels_hc_Mid == l, 1], 
                s=.5, # marker size
                alpha=0.5, # transparency
                label='Cluster %s' % l, # label
                )
plt.legend(bbox_to_anchor=(1,1), loc="upper left", markerscale = 20)
plt.xlabel('UMAP_0')
plt.ylabel('UMAP_1') 
plt.show()


unique_labels, lanel_cnts = np.unique(labels_spectral, return_counts=True)
for l in unique_labels:  
    plt.scatter(X_display_UMAP[labels_spectral == l, 0], X_display_UMAP[labels_spectral == l, 1], 
                s=.5, # marker size
                alpha=0.5, # transparency
                label='Cluster %s' % l, # label
                )
plt.legend(bbox_to_anchor=(1,1), loc="upper left", markerscale = 20)
plt.xlabel('UMAP_0')
plt.ylabel('UMAP_1')
plt.show()

#%% Output

cluster_name = 'Spectral'

d = {'Name':all_data['hemo-nucleus_Name'],'Cluster_Labels':labels_spectral}

df = pd.DataFrame(d)
df.to_csv(f'{save_path}/Result_clustering_{cluster_name}.csv')



cluster_name = 'Low'

#rois = np.concatenate(df_roi_list,axis=0)

d = {'Name':all_data['hemo-nucleus_Name'],'Cluster_Labels':labels_hc_Low}

df = pd.DataFrame(d)
df.to_csv(f'{save_path}/Result_clustering_{cluster_name}.csv')


#######

cluster_name = 'Mid'

d = {'Name':all_data['hemo-nucleus_Name'],'Cluster_Labels':labels_hc_Mid}

df = pd.DataFrame(d)
df.to_csv(f'{save_path}/Result_clustering_{cluster_name}.csv')


############

cluster_name = 'High'

d = {'Name':all_data['hemo-nucleus_Name'],'Cluster_Labels':labels_hc_High}

df = pd.DataFrame(d)
df.to_csv(f'{save_path}/Result_clustering_{cluster_name}.csv')




#%% Superrvised Learing to Calculate Feature Importance

labels_spectral = pd.DataFrame(labels_spectral)
labels_spectral.columns = ['HC_Clusters']

all_data.reset_index(drop=True, inplace=True)
labels_spectral.reset_index(drop=True, inplace=True)

all_data_with_clust = pd.concat([all_data, labels_spectral], axis = 1) 

feature_used = all_data_with_clust.drop(columns = ['hemo-nucleus_X', 'hemo-nucleus_Y',
                                                   'hemo-nucleus_XM', 'hemo-nucleus_YM', 
                                                   'hemo-nucleus_Name', 'HC_Clusters']).columns.tolist()

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



#%%

X_2 = all_data_with_clust.drop(columns = ['hemo-nucleus_X', 'hemo-nucleus_Y',
                                                   'hemo-nucleus_XM', 'hemo-nucleus_YM', 
                                                   'hemo-nucleus_Name']).values
y = all_data_with_clust['HC_Clusters'].values
X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size=0.30, random_state=0)


## Impute missing data

imputer = SimpleImputer(missing_values=np.nan,
                        strategy='constant', fill_value = 0)
imputer.fit(X_train)

X_train = np.array(imputer.transform(X_train), dtype=np.float32)
X_test = np.array(imputer.transform(X_test), dtype=np.float32)




clf = RandomForestClassifier(max_depth=5, random_state=0)

clf.fit(X_train, y_train)

y_pred_grid = clf.predict(X_test)
print(classification_report(y_test, y_pred_grid))


#scorer = metrics.make_scorer(metrics.f1_score, average = 'weighted')




#Calculate Feature Importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
indices_top20 = indices[0:40]
top20_features_used = []

for f in range(0,40):
    a = feature_used[indices_top20[f]-1]
    top20_features_used.append(a)


# print("Feature importances ranking:")
# for f in range(0,19):
#     print(f)
#     print('{0:.2f}%'.format(importances[indices_top20[f]]*100).rjust(6, ' '),
#           'feature %d: %s ' % (indices_top20[f], top20_features_used[indices_top20[f]]))


# Plot the feature importances of the forest
plt.figure(figsize=(10, 8))
plt.ylabel("Feature importances")
plt.bar(range(len(top20_features_used)), importances[indices_top20], align="center")
plt.xticks(range(len(top20_features_used)), top20_features_used, rotation=90)
plt.show()

