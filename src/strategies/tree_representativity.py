from sklearn.tree import DecisionTreeRegressor
from collections import Counter
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import copy
import scipy
from math import dist
from collections import Counter, defaultdict
from scipy.cluster.vq import vq
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
import random
import itertools
from bisect import bisect

def choices(population, weights=None, *, cum_weights=None, k=1):
    """Copy of source code for random.choices added to random module in 3.6

    Return a k sized list of population elements chosen with replacement.

    If the relative weights or cumulative weights are not specified,
    the selections are made with equal probability.

    """
    if len(population) == 0:
        raise ValueError('Population cannot be empty')
    if cum_weights is None:
        if weights is None:
            total = len(population)
            return [population[int(random.random() * total)] for i in range(k)]
        cum_weights = list(itertools.accumulate(weights))
    elif weights is not None:
        raise TypeError('Cannot specify both weights and cumulative weights')
    if len(cum_weights) != len(population):
        raise ValueError('The number of weights does not match the population')
    total = cum_weights[-1]
    return [population[bisect(cum_weights, random.random() * total)] for i in range(k)]

def unbiased_var(label_list):
    n = len(label_list)
    if n < 2:
        return 0

    mean = sum(label_list)/n
    tot = 0
    for val in label_list:
        tot += (mean - val)**2

    return tot/(n-1)


def ClusterIndices(clustNum, labels_array): 
            return np.where(labels_array == clustNum)[0]

class Breiman_Tree_representativity:   #Breiman tree refers to standard regression tree, using representativity based criteria in the leaves


    def __init__(self, min_samples_leaf=None, seed=None):

        self.points = None
        self.labels = None
        self.labelled_indices = None
        self._num_points = 0
        self._num_labelled = 0

        if seed is None:
            self.seed = 0
        else:
            self.seed = seed

        if min_samples_leaf is None:
            self.min_samples_leaf=1
        else:
            self.min_samples_leaf=min_samples_leaf

        self.tree = DecisionTreeRegressor(random_state=self.seed,min_samples_leaf=self.min_samples_leaf)
        self._leaf_indices = []
        self._leaf_marginal = []
        self._leaf_var = []
        self._al_proportions =[]

        self._leaf_statistics_up_to_date = False
        self._leaf_proportions_up_to_date = False

        self._verbose = False

    def input_data(self, all_data, labelled_indices, labels, copy_data=True):
    
        if copy_data:
            all_data = copy.deepcopy(all_data)
            labelled_indices = copy.deepcopy(labelled_indices)
            labels = copy.deepcopy(labels)

        if len(all_data) < len(labelled_indices):
            raise ValueError('Cannot have more labelled indicies than points')

        if len(labelled_indices) != len(labels):
            raise ValueError('Labelled indicies list and labels list must be same length')

        if str(type(all_data)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting all_data to list of lists internally')
            all_data = all_data.tolist()

        if str(type(labelled_indices)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting labelled_indices to list internally')
            labelled_indices = labelled_indices.tolist()

        if str(type(labels)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting labels to list internally')
            labels = labels.tolist()

        self.points = all_data
        self._num_points = len(self.points)
        self._num_labelled = len(labels)

        # Get the labels

        temp = [None] * self._num_points
        for i,ind in enumerate(labelled_indices):
            temp[ind] = labels[i]
        self.labels = temp
        self.labelled_indices = list(labelled_indices)

    def fit_tree(self):
        self.tree.fit(np.array(self.points)[self.labelled_indices,:], 
            np.array(self.labels)[self.labelled_indices])
        self._leaf_indices = self.tree.apply(np.array(self.points)) #return index of leaf for each point
        self._leaf_statistics_up_to_date = False
        
    def get_depth(self):
        return(self.tree.get_n_leaves())

    def label_point(self, index, value):

        if self.labels is None:
            raise RuntimeError('No data in the tree')

        if len(self.labels) <= index:
            raise ValueError('Index {} larger than size of data in tree'.format(index))

        value = copy.copy(value)
        index = copy.copy(index)

        self.labels[index] = value
        self.labelled_indices.append(index)
        self._num_labelled += 1

    def predict(self, new_points):
        return(self.tree.predict(new_points))
    
    def calculate_leaf_statistics(self):
        temp = Counter(self._leaf_indices) #get the no. of points in different leaves thus density
        self._leaf_marginal = []
        self._leaf_var = []
        for key in np.unique(self._leaf_indices):
            self._leaf_marginal.append(temp[key]/self._num_points)  #proportion of each leaf
            temp_ind = [i for i,x in enumerate(self._leaf_indices) if x == key]
            #temp_labels = [x for x in self.labels if x is not None]
            temp_labels = [x for i,x in enumerate(self.labels) if x is not None and self._leaf_indices[i]==key]
            self._leaf_var.append(unbiased_var(temp_labels))
        self._leaf_statistics_up_to_date = True

    def al_calculate_leaf_proportions(self):
        if not self._leaf_statistics_up_to_date:
            self.calculate_leaf_statistics()
        al_proportions = []
        for i, val in enumerate(self._leaf_var):
            al_proportions.append(np.sqrt(self._leaf_var[i] * self._leaf_marginal[i]))
        al_proportions = np.array(al_proportions)/sum(al_proportions)
        self._al_proportions = al_proportions
        self._leaf_proportions_up_to_date = True
                
        #pick new leaves with replacement
    def pick_new_points(self, num_samples = 1):
        if not self._leaf_proportions_up_to_date:
            self.al_calculate_leaf_proportions()

        temp = Counter(np.array(self._leaf_indices)[[x for x in range(self._num_points
            ) if self.labels[x] is None]])
        point_proportions = {}
        for i,key in enumerate(np.unique(self._leaf_indices)):
            point_proportions[key] = self._al_proportions[i] / max(1,temp[key]) 
        temp_probs = np.array([point_proportions[key] for key in self._leaf_indices])
        temp_probs[self.labelled_indices] = 0
        temp_probs = temp_probs / sum(temp_probs)
        # print(sum(temp_probs))
        # print(f'temp_probs: {temp_probs}')
        # print(f'leaf_indices: {self._leaf_indices}')
        # print(f'al_proportions: {self._al_proportions}')
        # print(f'temp : {temp}')
        leaves_to_sample = np.random.choice(self._leaf_indices,num_samples, 
            p=temp_probs, replace = False) 
        
        
        
        points_to_label = []
        
        centroid=[]
        centroid_temp = []
        
        num_cluster = []
                
        new_cent = []
        
        last_leaf_end = 0
        
        number_cluster_per_leaf = []
        
        rx = []
        
        data_clust = []
        len_data_clust = []
        data_clust_index = []
        
        for leaf_ind,leaf in enumerate(np.unique(leaves_to_sample)):

            data_labeled_all = [x for i,x in enumerate(self.points)
                     if self.labels[i] is not None ]                                  # set of X all labeled in the tree
            
            data_leaf_all = [x for i,x in enumerate(self.points)
                     if self._leaf_indices[i] ==leaf]                                 # set of all X in leaf leaf
              
            data_leaf_all_index = [x for i,x in enumerate(range(self._num_points))
                     if self._leaf_indices[i] ==leaf]                                 # set of all X indices in leaf leaf
            
            data_leaf = [x for i,x in enumerate(self.points)
                     if self._leaf_indices[i] ==leaf and self.labels[i] is None ]     # set of X all unlabled in leaf leaf
            
            data_index = [x for i,x in enumerate(range(self._num_points)
                    ) if self._leaf_indices[i] ==leaf and self.labels[i] is None ]    # set of X_index all unlabled in leaf leaf
            
            data_index_lab = [x for i,x in enumerate(range(self._num_points)
                    ) if self._leaf_indices[i] ==leaf and self.labels[i] is not None ]  # set of X_index all labeled in leaf leaf
            
            data_lab = [x for i,x in enumerate(self.points)
                     if self._leaf_indices[i] ==leaf and self.labels[i] is not None ]   # set of X all labeled in leaf leaf
                        
            # K means 
            centroid = []
            kmeans = KMeans(n_clusters = Counter(leaves_to_sample)[leaf]).fit(data_leaf)     #cluster all unlabeled sample in the leaf
            centroid = np.append(centroid, kmeans.cluster_centers_)
                    
            centroid_temp = np.append(centroid_temp,kmeans.cluster_centers_)
                            
            centroid = np.hsplit(centroid_temp,len(centroid_temp)/int(len(data_leaf[0]))) 
            
            # adapt representativity
            for num in range (len(kmeans.cluster_centers_)):   
                
                num_cluster = np.append(num_cluster,len(ClusterIndices(num, kmeans.labels_)))    # no of samples in each cluster
            number_cluster_per_leaf = np.append(number_cluster_per_leaf,len(num_cluster))
            number_cluster_per_leaf = number_cluster_per_leaf.astype(int)
            num_cluster = num_cluster.astype(int)
            
            closest = []    
            for i in range(len(kmeans.cluster_centers_)):   #new clusters only
                data_cluster = []
                for j in range(int(num_cluster[i+last_leaf_end])):
                    
                    data_cluster = np.append(data_cluster,data_leaf[ClusterIndices(i,  kmeans.labels_)[j]])
                    data_cluster = np.hsplit(data_cluster,len(data_cluster)/int(len(data_leaf[0])))        # samples in cluster i
                    
                    data_clust = np.append(data_clust,data_leaf[ClusterIndices(i,  kmeans.labels_)[j]])
                    
                    data_clust_index = np.append(data_clust_index,data_index[ClusterIndices(i,  kmeans.labels_)[j]])
                    
                    data_clust = np.hsplit(data_clust,len(data_clust)/int(len(data_leaf[0])))   # samples in cluster i
                len_data_clust = np.append(len_data_clust,len(data_clust))
                
                    
                centroid[i+last_leaf_end] = np.reshape(centroid[i+last_leaf_end],(1,-1))
                closest_ind, distances = vq(centroid[i+last_leaf_end], data_cluster)            # closest = indices of X_train_cluster i closest to centroid i
                closest = np.append(closest, ClusterIndices(i,  kmeans.labels_)[closest_ind])
                            
            for s in closest:
                new_cent = np.append(new_cent,data_leaf[int(s)])
            new_cent = np.hsplit(new_cent,len(new_cent)/int(len(data_leaf[0])))
            
            for p in range(len(kmeans.cluster_centers_)):   # for every cluster except the fixed one
                rxx=np.zeros(num_cluster[p+last_leaf_end])
                for q in range(num_cluster[p+last_leaf_end]):    # for every sample in the cluster

                    for k in range(num_cluster[p+last_leaf_end]):

                        rxx[q]+= abs(dist(data_leaf[ClusterIndices(p, kmeans.labels_)[k]],data_leaf[ClusterIndices(p, kmeans.labels_)[q]]))
                    rxx[q] = rxx[q]/(num_cluster[p+last_leaf_end]-1) 
                rx.append(rxx)   # average value of R for clusters
            
            last_leaf_end = len(num_cluster)
            
        # outside loop 

        # calculate D
            
        c2 = []
        cmax=2
        len_data_clust = len_data_clust.astype(int)
        data_in_cluster = []
        for u in range (cmax):   # cmax = # of optimisations
            c1=[]
            data_in_cluster = data_clust[0:len_data_clust[0]]
            
            for p in range(len(new_cent)):   # for each cluster in the tree
                d = []
                for n in range(num_cluster[p]):   # for every sample in the cluster
                    #d_gsx = np.zeros(len(closest))
                    d_gsx = []

                    for l,m in enumerate(new_cent):   # distance to every other centroid
                        #if closest[p]!=m:
                        if p!=l:
                            d_gsx = np.append(d_gsx,dist(new_cent[l],data_in_cluster[n]))

                    if 0 in d_gsx:    
                        return(d_gsx,num_cluster,closest,centroid) 

                    for h in range(len(data_labeled_all)):                      # get the distance to all the labeled samples
                                d_gsx = np.append(d_gsx,dist(data_labeled_all[h],data_in_cluster[n]))

                    d = np.append(d,np.amin(d_gsx))              
            
                c= np.argmax(d-rx[p])
                new_cent[p] = data_in_cluster[c]
                
                c0= np.argmax(d-rx[p])
                if p>0:
                    c0+=len_data_clust[p-1]
                c1 = np.append(c1,data_clust_index[c0])
                
                if len(c1) != len(new_cent):
                    data_in_cluster = data_clust[len_data_clust[p]:len_data_clust[p+1]]
                
            c2.append(c1)

            if u>0 and np.all(c2[u] == c2[u-1]) or u==cmax-1:
                c2[u] = np.array(c2[u])
                c2[u] = c2[u].astype(int)
                for c2_idx, irdm_idx in enumerate(c2[u]):
                    points_to_label.append(irdm_idx)
                break

        return(points_to_label)
