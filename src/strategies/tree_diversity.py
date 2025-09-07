from sklearn.tree import DecisionTreeRegressor
from collections import Counter
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import copy
import scipy
from math import dist
from sklearn.tree import export_text
import sklearn

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

class Breiman_Tree_diversity:    #Breiman tree refers to a standard regression tree, using diversity-based criteria in the leaves

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

        #List of labels
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
    
    def export_text(self):
        return(sklearn.tree.export_text(self.tree))
        
        
    #Function to calculate pi and sigma
    
    def calculate_leaf_statistics(self):
        temp = Counter(self._leaf_indices)      #get the no. of samples in different leaves thus density
        self._leaf_marginal = []
        self._leaf_var = []
        for key in np.unique(self._leaf_indices):
            self._leaf_marginal.append(temp[key]/self._num_points)  #proportion of each leaf
            temp_ind = [i for i,x in enumerate(self._leaf_indices) if x == key]
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
        leaves_to_sample = np.random.choice(self._leaf_indices,num_samples, 
            p=temp_probs, replace = False)     #leaves to be sampled from have been selected'''
        
        
        #Label the samples based on RT(Diversity)
        
        points_to_label = []
        
        for leaf in np.unique(leaves_to_sample):
            points = []
            
            
            data_labeled_all = [x for i,x in enumerate(self.points)
                     if self.labels[i] is not None ]                                 # set of X all labeled in the tree
            
            data_leaf_all = [x for i,x in enumerate(self.points)
                     if self._leaf_indices[i] ==leaf]                                #set of all X in leaf leaf
             
            data_leaf_all_index = [x for i,x in enumerate(range(self._num_points))
                     if self._leaf_indices[i] ==leaf]                                #set of all X indices in leaf leaf
              
            data_leaf = [x for i,x in enumerate(self.points)
                     if self._leaf_indices[i] ==leaf and self.labels[i] is None ]    # set of X all unlabled in leaf leaf
            
            data_index = [x for i,x in enumerate(range(self._num_points)
                    ) if self._leaf_indices[i] ==leaf and self.labels[i] is None ]   # set of X_index all unlabled in leaf leaf
            
            data_index_lab = [x for i,x in enumerate(range(self._num_points)
                    ) if self._leaf_indices[i] ==leaf and self.labels[i] is not None ]  # set of X_index all labeled in leaf leaf
            
            data_lab = [x for i,x in enumerate(self.points)
                     if self._leaf_indices[i] ==leaf and self.labels[i] is not None ]   # set of X all labeled in leaf leaf
            
            
            if len(data_leaf)>0:
                            
                dist_mat_lab_unlab = scipy.spatial.distance_matrix(data_labeled_all, data_leaf)

                z = np.zeros(len(dist_mat_lab_unlab))
                for i in range(len(dist_mat_lab_unlab)):
                    z[i]=sum(dist_mat_lab_unlab[i])


                D2 = dist_mat_lab_unlab[np.argmin(z)]     #vector of distances of all labeled in the tree closest to all unlabeled in the leaf 

                first_point_to_be_labeled = np.argmax(D2)    #index from unlabeled list

                points_to_label.append(data_index[first_point_to_be_labeled])  
            
            
            if len(data_leaf)>1 and Counter(leaves_to_sample)[leaf]>1:
            
            
                dist_mat = scipy.spatial.distance_matrix(data_leaf, data_leaf)   #distance matrix of training set, unlabeled            

                all_ind = [i for i in range(0,len(dist_mat))]   
                labeled_ind = [first_point_to_be_labeled]
                K=0    
                for j in range(Counter(leaves_to_sample)[leaf] - 1):     #-1 as one sample has already been labeled

                    unlab_ind=np.setdiff1d(all_ind,labeled_ind)
                    D2 = np.delete(D2,np.argmax(D2))   #remove last labeled point from D2
                    D=[]
                    D = [np.append(D,dist_mat[labeled_ind[len(labeled_ind)-1]][i]) for i in unlab_ind]  #dist of unlabeled to last labeled sample
                    D3 = np.append(D2,D)
                    D3=D3.reshape(2,-1)
                    D2 = [D3[:,i].min() for i in range(len(D3[0]))]
                    new_ind = unlab_ind[np.argmax(D2)]
                    labeled_ind = np.append(labeled_ind,new_ind)

                    points_to_label.append(data_index[new_ind])
                
        return(points_to_label)
