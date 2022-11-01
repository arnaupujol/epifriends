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
        Minium number of neighbours in the radius < link_d needed to link cases
        as friends

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

def catalogue(positions, test_result, link_d, cluster_id = None, \
                min_neighbours = 2, max_p = 1, min_pos = 2, min_total = 2, \
                min_pr = 0):
    """
    This method runs the DBSCAN algorithm (if cluster_id is None) and obtains the mean
    positivity rate (PR) of each cluster extended with the non-infected cases
    closer than the link_d.

    Parameters:
    -----------
    positions: np.ndarray
        An array with the position parameters with shape (n,2),
        where n is the number of positions
    test_result: np.array
        An array with the test results (0 or 1)
    link_d: float
        The linking distance to connect cases
    cluster_id: np.array
        An array with the cluster ids of the positive cases
    min_neighbours: int
        Minium number of neighbours in the radius < link_d needed to link cases
        as friends
    max_p: float
        Maximum value of the p-value to consider the cluster detection
    min_pos: int
        Threshold of minimum number of positive cases in clusters applied
    min_total: int
        Threshold of minimum number of cases in clusters applied
    min_pr: float
        Threshold of minimum positivity rate in clusters applied

    Returns:
    --------
    cluster_id: np.array
        List of the cluster IDs of each position, with 0 for those
        without a cluster.
    mean_pr_fof: np.array
        Mean PR corresponding to cluster_id
    pval_fof: np.array
        P-value corresponding to cluster_id
    epifriends_catalogue: geopandas.DataFrame
        Catalogue of the epifriends clusters and their main characteristics
    """
    #Define positions of positive cases
    positive_positions = positions[test_result == 1]
    #Computing cluster_id if needed
    if cluster_id is None:
        cluster_id = dbscan(positive_positions, link_d, \
                            min_neighbours = min_neighbours)
    #Create KDTree for all populations
    tree =spatial.KDTree(positions)
    #Define total number of positive cases
    total_positives = np.sum(test_result)
    #Define total number of cases
    total_n = len(test_result)

    #Initialising mean PR and p-value for the positive cases in clusters
    mean_pr_cluster = np.zeros_like(cluster_id)
    pval_cluster = np.ones_like(cluster_id)
    #EpiFRIenDs cluster catalogue
    epifriends_catalogue = {'id' : [], #EpiFRIenDs id
                     'mean_position_pos' : [], #Mean position of positive cases
                     'mean_position_all' : [], #Mean position of all cases
                     'mean_pr' : [], #Positivity rate
                     'positives' : [], #Number of positive cases
                     'negatives' : [], #Number of negative cases
                     'total' : [], #Total number of positions
                     'indeces' : [], #Indeces of all positions
                     'p' : [], #p-value of detection
                    }
    next_id = 1
    for i,f in enumerate(np.unique(cluster_id[cluster_id>0])):
        #get all indeces with this cluster id
        has_this_cluster_id = cluster_id == f
        cluster_id_indeces = np.arange(len(positive_positions))[has_this_cluster_id]
        #for all these indeces, get list of friends from all positions
        all_friends_indeces = find_indeces(positive_positions[cluster_id_indeces], link_d, tree)
        #get unique values of such indeces
        total_friends_indeces = np.unique(np.concatenate(all_friends_indeces))
        #get positivity rate from all the unique indeces
        mean_pr = np.mean(test_result[total_friends_indeces])
        npos = np.sum(test_result[total_friends_indeces])
        ntotal = len(total_friends_indeces)
        pval = 1 - stats.binom.cdf(npos - 1, ntotal, \
                                    total_positives/total_n)

        #setting EpiFRIenDs catalogue
        if pval < max_p and npos >= min_pos and ntotal >= min_total and \
                mean_pr >= min_pr:
            epifriends_catalogue['id'].append(next_id)
            cluster_id[cluster_id_indeces] = next_id
            next_id+=1

            mean_pr_cluster[cluster_id_indeces] = mean_pr
            pval_cluster[cluster_id_indeces] = pval

            mean_pos = np.mean(positive_positions[cluster_id_indeces], axis = 0)
            epifriends_catalogue['mean_position_pos'].append(mean_pos)
            mean_pos_ext = np.mean(positions[total_friends_indeces], axis = 0)
            epifriends_catalogue['mean_position_all'].append(mean_pos_ext)
            epifriends_catalogue['mean_pr'].append(mean_pr)
            epifriends_catalogue['positives'].append(int(npos))
            epifriends_catalogue['negatives'].append(int(ntotal - npos))
            epifriends_catalogue['total'].append(int(ntotal))
            epifriends_catalogue['indeces'].append(total_friends_indeces)
            epifriends_catalogue['p'].append(pval)
        else:
            cluster_id[cluster_id_indeces] = 0
    #Make the epifriends_catalogue a geopandas dataframe
    epifriends_catalogue = dict2geodf(epifriends_catalogue)
    return cluster_id, mean_pr_cluster, pval_cluster, epifriends_catalogue

def dict2geodf(dict_catalogue, epsg = 3857):
    """
    This method transforms the EpiFRIenDs catalogue dictionary
    into a geopandas dataframe.

    Parameters:
    -----------
    dict_catalogue: dict
        Dictionary with the EpiFRIenDs catalogue
    epsg: int
        GIS spatial projection of coordinates

    Returns:
    --------
    geo_catalogue: geopandas.GeoDataFrame
        Data frame of catalogue with GIS coordinates
    """
    geo_catalogue = pd.DataFrame(dict_catalogue)
    x_points = np.array([i[0] for i in geo_catalogue['mean_position_pos']])
    y_points = np.array([i[1] for i in geo_catalogue['mean_position_pos']])
    geo_catalogue = geopandas.GeoDataFrame(geo_catalogue, \
                        geometry = geopandas.points_from_xy(x_points, y_points))
    geo_catalogue = geo_catalogue.set_crs(epsg=epsg)
    return geo_catalogue

def distance(pos_a, pos_b):
    """
    This method calculates the Euclidean distance between two positions.

    Parameters:
    -----------
    pos_a: np.ndarray
        First position
    pos_b: np.ndarray
        Second position

    Returns:
    --------
    dist: float
        Distance between positions
    """
    dist = np.sqrt(np.sum((pos_a - pos_b)**2))
    return dist

def temporal_catalogue(positions, test_result, dates, link_d, min_neighbours, \
                       time_width, min_date, max_date, time_steps = 1, \
                       max_p = 1, min_pos = 2, min_total = 2, min_pr = 0):
    """
    This method generates a list of EpiFRIenDs catalogues representing different time frames
    by including only cases within a time window that moves within each time step.

    Parameters:
    -----------
    positions: np.ndarray
        An array with the position parameters with shape (n,2),
        where n is the number of positions
    test_result: np.array
        An array with the test results (0 or 1)
    dates: pd.Series or np.array
        List of the date times of the corresponding data
    link_d: float
        The linking distance to connect cases
    min_neighbours: int
        Minium number of neighbours in the radius < link_d needed to link cases
        as friends
    time_width: int
        Number of days of the time window used to select cases in each time step
    min_date: pd.DateTimeIndex
        Initial date used in the first time step and time window selection
    max_date: pd.DateTimeIndex
        Final date to analyse, defining the last time window as the one fully overlapping
        the data
    time_steps: int
        Number of days that the time window is shifted in each time step
    max_p: float
        Maximum value of the p-value to consider the cluster detection
    min_pos: int
        Threshold of minimum number of positive cases in clusters applied
    min_total: int
        Threshold of minimum number of cases in clusters applied
    min_pr: float
        Threshold of minimum positivity rate in clusters applied

    Returns:
    --------
    temporal_catalogues: list of pandas.DataFrame
        List of EpiFRIenDs catalogues, where each element contains the catalogue in each
        time step
    mean_date: list
        List of dates corresponding to the median time in each time window
    """
    #Defining temporal range
    dates = pd.to_datetime(dates)
    if min_date is None:
        min_date = dates.min()
    if max_date is None:
        max_date = dates.max()
    #temporal loop until the last time frame that fully overlaps the data
    temporal_catalogues = []
    #Mean dates defines as the median time in each time window
    mean_date = []
    step_num = 0
    while min_date + pd.to_timedelta(time_steps*step_num + time_width, unit = 'D') <= max_date:
        #select data in time window
        selected_data = (dates >= min_date + pd.to_timedelta(time_steps*step_num, unit = 'D'))& \
                        (dates <= min_date + pd.to_timedelta(time_steps*step_num + time_width, unit = 'D'))
        selected_positions = positions[selected_data]
        selected_test_results = test_result[selected_data]

        #get catalogue
        cluster_id, mean_pr_cluster, pval_cluster, \
        epifriends_catalogue = catalogue(selected_positions, selected_test_results, \
                                         link_d, min_neighbours = min_neighbours, \
                                         max_p = max_p, min_pos = min_pos, \
                                         min_total = min_total, min_pr = min_pr)
        #get median date
        mean_date.append(min_date + pd.to_timedelta(time_steps*step_num + .5*time_width, unit = 'D'))#TODO test
        epifriends_catalogue['Date'] = mean_date[-1]
        temporal_catalogues.append(epifriends_catalogue)

        step_num +=1
    return temporal_catalogues, mean_date

def add_temporal_id(catalogue_list, linking_time, linking_dist, \
                    get_timelife = True):
    """
    This method generates the temporal ID of EpiFRIenDs clusters by linking
    clusters from different time frames, assigning the same temporal ID to
    them when they are close enough in time and space.

    Parameters:
    -----------
    catalogue_list: list of pandas.DataFrame
        List of EpiFRIenDs catalogues, each element of the list
        corresponding to the catalogue of each timestep
    linking_time: int
        Maximum number of timesteps of distance to link hotspots with
        the same temporal ID
    linking_dist: float
        Linking distance used to link the clusters from the different
        time frames
    get_timelife: bool
        It specifies if the time periods and timelife of clusters are obtained

    Returns:
    --------
    catalogue_list: list of pandas.DataFrame
        List of EpiFRIenDs catalogues with the added variable 'tempID' (and
        optionally the variables 'first_timestep', 'last_timestep' and
        'lifetime')
    """
    #setting empty values of temp_id
    for t in range(len(catalogue_list)):
        catalogue_list[t]['tempID'] = pd.Series(dtype = int)
    #Initialising tempID value to assign
    next_temp_id = 0
    #Loop over all timesteps
    for t in range(len(catalogue_list)):
        #Loop over all clusters in a timestep
        for f in catalogue_list[t].T:
            #Loop over all timesteps within linking_time
            for t2 in range(t + 1, min(t + linking_time + 1, len(catalogue_list))):
                #Loop over all clusters in the linked timesteps
                for f2 in catalogue_list[t2].T:
                    #Calculating distance between clusters
                    dist = distance(catalogue_list[t].loc[f]['mean_position_pos'], \
                                    catalogue_list[t2].loc[f2]['mean_position_pos'])
                    if dist <= linking_dist:
                        temp_id1 = catalogue_list[t].loc[f]['tempID']
                        temp_id2 = catalogue_list[t2].loc[f2]['tempID']
                        #Assign tempIDs to linked clusters
                        if np.isnan(temp_id1) and np.isnan(temp_id2):
                            catalogue_list[t]['tempID'].loc[f] = next_temp_id
                            catalogue_list[t2]['tempID'].loc[f2] = next_temp_id
                            next_temp_id += 1
                        elif np.isnan(temp_id1):
                            catalogue_list[t]['tempID'].loc[f] = temp_id2
                        elif np.isnan(temp_id2):
                            catalogue_list[t2]['tempID'].loc[f2] = temp_id1
                        elif temp_id1 != temp_id2:
                            for t3 in range(len(catalogue_list)):
                                catalogue_list[t3]['tempID'].loc[catalogue_list[t3]['tempID'] == temp_id2] = temp_id1
    if get_timelife:
        catalogue_list = get_lifetimes(catalogue_list)
    return catalogue_list

def get_lifetimes(catalogue_list):
    """
    This method obtains the first and last time frames for each
    temporal ID from a list of EpiFRIenDs catalogues and the corresponding
    timelife.

    Parameters:
    -----------
    catalogue_list: list of pandas.DataFrame
        List of EpiFRIenDs catalogues, each element of the list
        corresponding to the EpiFRIenDs catalogue of each timestep

    Returns:
    --------
    catalogue_list: list of pandas.DataFrame
        List of hotspot catalogues with the added fields 'first_timestep',
        'last_timestep' and 'lifetime'
    """
    #getting list of temporal IDs appearing in catalogue_list
    tempid_list = get_label_list(catalogue_list, label = 'tempID')
    #Creating empty columns for first timestep, last timestep and lifteime
    for t in range(len(catalogue_list)):
            catalogue_list[t]['first_timestep'] = pd.Series(dtype = int)
            catalogue_list[t]['last_timestep'] = pd.Series(dtype = int)
            catalogue_list[t]['lifetime'] = 0
    for tempid_num in tempid_list:
        appearances = []
        for i in range(len(catalogue_list)):
            if tempid_num in catalogue_list[i]['tempID'].unique():
                appearances.append(i)
        min_appearance = min(appearances)
        max_appearance = max(appearances)
        lifetime = max_appearance - min_appearance
        for i in range(min_appearance, max_appearance + 1):
            catalogue_list[i]['first_timestep'].loc[catalogue_list[i]['tempID'] == tempid_num] = min_appearance
            catalogue_list[i]['last_timestep'].loc[catalogue_list[i]['tempID'] == tempid_num] = max_appearance
            catalogue_list[i]['lifetime'].loc[catalogue_list[i]['tempID'] == tempid_num] = lifetime
    return catalogue_list

def get_label_list(df_list, label = 'tempID'):
    """
    This method gives the unique values of a column in a list
    of data frames.

    Parameters:
    -----------
    df_list: list of pandas.DataFrames
        List of dataframes
    label: str
        Name of column to select

    Returns:
    --------
    label_list: list
        List of unique values of the column over all dataframes from the list
    """
    for i in range(len(df_list)):
        mask = df_list[i][label].notnull()
        if i == 0:
            label_list = df_list[i][label].loc[mask].unique()
        else:
            label_list = np.unique(np.concatenate((label_list, df_list[i][label].loc[mask].unique())))
    return label_list
