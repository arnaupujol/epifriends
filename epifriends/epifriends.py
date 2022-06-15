#This module contains methods to identify EpiFRIenDs clusters.

import numpy as np
import pandas as pd
import geopandas
from scipy import spatial, stats

def find_indeces(positions, link_d, tree):
    """
    This method returns the indeces of all the friends
    of each position from positions given a KDTree.

    Parameters:
    -----------
    positions: np.ndarray
        An array with the position parameters with shape (n,2),
        where n is the number of positions
    link_d: float
        The linking distance to label friends
    tree: scipy.spatial.KDTree
        A KDTree build from the positions of the target data

    Returns:
    --------
    indeces: list
        List with an array of the indeces of the friends of each
        position
    """
    indeces = []
    for i in range(len(positions)):
        indeces.append([])
        dist = 0
        kth = 0
        while dist <= link_d:
            kth += 1
            dist, index = tree.query([positions[i]], k = [kth])
            if dist == 0 and kth > 1:#avoiding issue for >1 point with dist == 0
                d, index = tree.query([positions[i]], k = kth)
                indeces[i] = index[0].tolist()
            elif dist <= link_d:
                indeces[i].append(index[0][0])
            else:
                break
        indeces[i] = np.array(indeces[i], dtype = int)
    return indeces

def dbscan(positions, link_d, min_neighbours = 2):
    """
    This method finds the DBSCAN clusters from a set of positions and
    returns their cluster IDs.

    Parameters:
    -----------
    positions: np.ndarray
        An array with the position parameters with shape (n,2),
        where n is the number of positions
    link_d: float
        The linking distance of the DBSCAN algorithm
    min_neighbours: int
        Minium number of neighbours in the radius < link_d needed to consider
        the cluster

    Returns:
    --------
    cluster_id: np.array
        List of the cluster IDs of each position, with 0 for those
        without a cluster.
    """
    #Create cluster id
    cluster_id = np.zeros(len(positions))

    #Create KDTree
    tree =spatial.KDTree(positions)
    #Query KDTree
    indeces = find_indeces(positions, link_d, tree)

    last_cluster_id = 0
    for i in range(len(positions)):
        #check if ith position has any neighbour
        if len(indeces[i]) < min_neighbours:
            continue
        else:
            #Define indeces of selected friends
            indeces_friends = indeces[i]
            #cluster_ids of these friends
            cluster_id_friends = cluster_id[indeces_friends]
            #Unique values of cluster_ids
            unique_cluster_ids = np.unique(cluster_id_friends)
            #check values of cluster_id in these neighbours
            if len(unique_cluster_ids) == 1:
                if unique_cluster_ids[0] == 0:
                    #assign to ith and friends last_cluster_id
                    cluster_id[indeces_friends] = last_cluster_id + 1
                    last_cluster_id+=1
                else:
                    #if one cluster_id different than 0, assign it to ith and friends
                    cluster_id[indeces_friends] = unique_cluster_ids[0]
            else:
                #Define the cluster_id to assign for merging several clusters
                min_cluster_id = np.min(unique_cluster_ids[unique_cluster_ids != 0])
                #Assign this cluster_id to ith and its friends
                cluster_id[indeces_friends] = min_cluster_id
                #Assign it to all cases with any of these cluster_id_friends
                for j in unique_cluster_ids[unique_cluster_ids != 0]:
                    cluster_id[cluster_id == j] = min_cluster_id
    #Rename cluster_id to continuous integers
    for i, f in enumerate(np.unique(cluster_id[cluster_id>0])):
        cluster_id[cluster_id == f] = i+1
    return cluster_id
