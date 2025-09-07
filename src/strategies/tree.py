#!/usr/bin/env python3
"""
Breiman Tree Active Learning Strategy

This module implements tree-based active learning strategies using regression trees
with uncertainty-based sample selection for active learning.

Author: Jinghou Bi
Email: jinghou.bi@tu-dresden.de
License: MIT License
Created: 2025-01-06
"""

from sklearn.tree import DecisionTreeRegressor
from collections import Counter

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import copy

import random
import itertools
from bisect import bisect


def choices(population, weights=None, *, cum_weights=None, k=1):
    """
    Copy of source code for random.choices added to random module in 3.6.

    Return a k sized list of population elements chosen with replacement.

    Args:
        population (list): Population to choose from.
        weights (list, optional): Relative weights for selection.
        cum_weights (list, optional): Cumulative weights for selection.
        k (int, optional): Number of samples to select. Defaults to 1.

    Returns:
        list: Selected population elements.
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
    """
    Calculate unbiased variance of a label list.

    Args:
        label_list (list): List of labels.

    Returns:
        float: Unbiased variance.
    """
    n = len(label_list)
    if n < 2:
        return 0

    mean = sum(label_list)/n
    tot = 0
    for val in label_list:
        tot += (mean - val)**2

    return tot/(n-1)


class Breiman_Tree:
    """
    Breiman tree refers to a standard regression tree.

    This class implements tree-based active learning using regression trees
    with uncertainty-based sample selection.

    Attributes:
        min_samples_leaf (int): Minimum samples required at leaf nodes.
        seed (int): Random seed for reproducibility.
        tree (DecisionTreeRegressor): The underlying decision tree.
    """

    def __init__(self, min_samples_leaf=None, seed=None):
        """
        Initialize the Breiman Tree strategy.

        Args:
            min_samples_leaf (int, optional): Minimum samples per leaf. Defaults to 1.
            seed (int, optional): Random seed. Defaults to 0.
        """
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
        """
        Input data for the Breiman Tree.

        Args:
            all_data (array-like): All data points.
            labelled_indices (array-like): Indices of labelled data points.
            labels (array-like): Labels for the labelled data points.
            copy_data (bool, optional): Whether to copy data internally. Defaults to True.

        Raises:
            ValueError: If there are more labelled indices than points, or if
                        labelled indices and labels have different lengths.
        """
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

        #Set the labels

        temp = [None] * self._num_points
        for i,ind in enumerate(labelled_indices):
            temp[ind] = labels[i]
        self.labels = temp
        self.labelled_indices = list(labelled_indices)

    def fit_tree(self):
        """
        Fit the regression tree to the labelled data.
        """
        self.tree.fit(np.array(self.points)[self.labelled_indices,:],
            np.array(self.labels)[self.labelled_indices])
        self._leaf_indices = self.tree.apply(np.array(self.points)) #return index of leaf for each point, labeled and unlabeled
        self._leaf_statistics_up_to_date = False
        
    def get_depth(self):
        """
        Get the depth of the tree.

        Returns:
            int: Depth of the tree.
        """
        return(self.tree.get_n_leaves())

    def label_point(self, index, value):
        """
        Label a specific point in the dataset.

        Args:
            index (int): Index of the point to label.
            value: Label value.

        Raises:
            RuntimeError: If there is no data in the tree.
            ValueError: If the index is out of bounds.
        """
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
        """
        Predict labels for new data points.

        Args:
            new_points (array-like): New data points to predict labels for.

        Returns:
            array-like: Predicted labels.
        """
        return(self.tree.predict(new_points))
        
        
    # Calculate pi and sigma

    def calculate_leaf_statistics(self):
        """
        Calculate leaf statistics (marginal, variance) for all leaves.
        """
        temp = Counter(self._leaf_indices)
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
        """
        Calculate alpha-level proportions for active learning.
        """
        if not self._leaf_statistics_up_to_date:
            self.calculate_leaf_statistics()
        al_proportions = []
        for i, val in enumerate(self._leaf_var):
            al_proportions.append(np.sqrt(self._leaf_var[i] * self._leaf_marginal[i]))
        al_proportions = np.array(al_proportions)/sum(al_proportions)
        self._al_proportions = al_proportions
        self._leaf_proportions_up_to_date = True
        
        
    #Pick new samples to be labeled using Random sampling

    def pick_new_points(self, num_samples = 1):   
        """
        Pick new samples to be labeled using Random sampling.

        Args:
            num_samples (int, optional): Number of samples to select. Defaults to 1.

        Returns:
            list: Indices of selected samples.
        """
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
        if 'NaN' in temp_probs:
            return(temp,temp_probs,sum(temp_probs))
        leaves_to_sample = np.random.choice(self._leaf_indices,num_samples, 
            p=temp_probs, replace = False)   
        
        #points to label randomly based on leaf proportions
        points_to_label = []
        for leaf in np.unique(leaves_to_sample):
            points = []
            for j in range(Counter(leaves_to_sample)[leaf]):
            
                possible_points = np.setdiff1d([x for i,x in enumerate(range(self._num_points)
                    ) if self._leaf_indices[i] ==leaf and self.labels[i] is None ], points)
                                               
                point_to_label = np.random.choice(possible_points)
                points_to_label.append(point_to_label)
                points.append(point_to_label)  

        return(points_to_label)
