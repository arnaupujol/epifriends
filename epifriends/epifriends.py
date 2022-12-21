#This module contains methods to identify EpiFRIenDs clusters.

import numpy as np
import pandas as pd
import geopandas
from scipy import spatial, stats
from epifriends.utils import clean_unknown_data, get_2dpositions, find_indeces
from epifriends.utils import dict2geodf, distance, get_label_list

def dbscan(x, y, link_d, min_neighbours = 2, in_latlon = False, to_epsg = None, \
           verbose = True):
    """
    This method finds the DBSCAN clusters from a set of positions and
    returns their cluster IDs.

    Parameters:
    -----------
    x: np.array
        Vector of x geographical positions
    y: np.array
        Vector of y geographical positions
    link_d: float
        The linking distance of the DBSCAN algorithm
    min_neighbours: int
        Minium number of neighbours in the radius < link_d needed to link cases
        as friends
    in_latlon: bool
        If True, x and y coordinates are treated as longitude and latitude
        respectively, otherwise they are treated as cartesian coordinates
    to_epsg: int
        If in_latlon is True, x and y are reprojected to this EPSG
    verbose: bool
        It specifies if extra information is printed in the process

    Returns:
    --------
    cluster_id: np.array
        List of the cluster IDs of each position, with 0 for those
        without a cluster.
    """
    #Removing elements with missing positions
    x, y = clean_unknown_data(x, y, verbose = verbose)
    #Defining 2d-positions
    positions = get_2dpositions(x, y, in_latlon = in_latlon, to_epsg = to_epsg, \
                                      verbose = verbose)
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

def catalogue(x, y, test_result, link_d, cluster_id = None, \
                min_neighbours = 2, max_p = 1, min_pos = 2, min_total = 2, \
                min_pr = 0, in_latlon = False, to_epsg = None, \
                keep_null_tests = True, verbose = True):
    """
    This method runs the DBSCAN algorithm (if cluster_id is None) and obtains
    the mean positivity rate (PR) of each cluster extended with the non-infected
    cases closer than the link_d.

    Parameters:
    -----------
    x: np.array
        Vector of x geographical positions
    y: np.array
        Vector of y geographical positions
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
    in_latlon: bool
        If True, x and y coordinates are treated as longitude and latitude
        respectively, otherwise they are treated as cartesian coordinates
    to_epsg: int
        If in_latlon is True, x and y are reprojected to this EPSG
    keep_null_tests: bool, int or float
        It defines how to treat the missing test results. If True, they are kept
        as missing, that will included foci, contributing to the total size and
        the p-value but not to the number of positives, negatives and
        positivity. If False, they are removed and not used. If int or float,
        the value is assigned to them, being interpreted as positive for 1 and
        negative for 0
    verbose: bool
        It specifies if extra information is printed in the process

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
    #Removing elements with missing positions
    x, y, test_result = clean_unknown_data(x, y, test = test_result, \
                                                 keep_null_tests = keep_null_tests, \
                                                 verbose = verbose)
    #Defining 2d-positions
    positions = get_2dpositions(x, y, in_latlon = in_latlon, to_epsg = to_epsg, \
                                      verbose = verbose)
    #Define positions of positive cases
    are_positive = test_result == 1
    positive_positions = positions[are_positive]
    #Computing cluster_id if needed
    if cluster_id is None:
        cluster_id = dbscan(positive_positions[:,0], positive_positions[:,1], \
                            link_d, min_neighbours = min_neighbours, \
                            in_latlon = False, verbose = False)
    #Create KDTree for all populations
    tree = spatial.KDTree(positions)
    #Define total number of positive cases
    total_positives = np.nansum(test_result)
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
        mean_pr = np.nanmean(test_result[total_friends_indeces])
        npos = np.nansum(test_result[total_friends_indeces])
        nneg = np.sum(test_result[total_friends_indeces] == 0)
        ntotal = len(total_friends_indeces)
        pval = 1 - stats.binom.cdf(npos - 1, ntotal, \
                                    total_positives/total_n)

        #setting EpiFRIenDs catalogue
        if pval <= max_p and npos >= min_pos and ntotal >= min_total and \
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
            epifriends_catalogue['negatives'].append(int(nneg))
            epifriends_catalogue['total'].append(int(ntotal))
            epifriends_catalogue['indeces'].append(total_friends_indeces)
            epifriends_catalogue['p'].append(pval)
        else:
            cluster_id[cluster_id_indeces] = 0
    #Make the epifriends_catalogue a geopandas dataframe
    epifriends_catalogue = dict2geodf(epifriends_catalogue)
    return cluster_id, mean_pr_cluster, pval_cluster, epifriends_catalogue

def temporal_catalogue(x, y, test_result, dates, link_d, min_neighbours = 2, \
                       time_width = 10, min_date = None, max_date = None, \
                       time_steps = 1, max_p = 1, min_pos = 2, min_total = 2, \
                       min_pr = 0, in_latlon = False, to_epsg = None, \
                       keep_null_tests = True, verbose = True):
    """
    This method generates a list of EpiFRIenDs catalogues representing different time frames
    by including only cases within a time window that moves within each time step.

    Parameters:
    -----------
    x: np.array
        Vector of x geographical positions
    y: np.array
        Vector of y geographical positions
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
    in_latlon: bool
        If True, x and y coordinates are treated as longitude and latitude
        respectively, otherwise they are treated as cartesian coordinates
    to_epsg: int
        If in_latlon is True, x and y are reprojected to this EPSG
    keep_null_tests: bool, int or float
        It defines how to treat the missing test results. If True, they are kept
        as missing, that will included foci, contributing to the total size and
        the p-value but not to the number of positives, negatives and
        positivity. If False, they are removed and not used. If int or float,
        the value is assigned to them, being interpreted as positive for 1 and
        negative for 0
    verbose: bool
        It specifies if extra information is printed in the process

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
        selected_x = x[selected_data]
        selected_y = y[selected_data]
        selected_test_results = test_result[selected_data]

        #get catalogue
        cluster_id, mean_pr_cluster, pval_cluster, \
        epifriends_catalogue = catalogue(selected_x, selected_y, selected_test_results, \
                                         link_d, min_neighbours = min_neighbours, \
                                         max_p = max_p, min_pos = min_pos, \
                                         min_total = min_total, min_pr = min_pr, \
                                         in_latlon = in_latlon, to_epsg = to_epsg, \
                                         keep_null_tests = keep_null_tests, \
                                         verbose = verbose)
        verbose = False
        #get median date
        mean_date.append(min_date + pd.to_timedelta(time_steps*step_num + .5*time_width, unit = 'D'))
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
    next_temp_id = 1
    #Loop over all timesteps
    for t in range(len(catalogue_list)):
        #Loop over all clusters in a timestep
        for f in catalogue_list[t].T:
            #Loop over all timesteps within linking_time
            for t2 in range(t + 1, min(t + linking_time + 1, len(catalogue_list))):
                #Loop over all clusters in the linked timesteps
                for f2 in catalogue_list[t2].T:
                    #Calculating distance between clusters
                    dist = distance(catalogue_list[t].loc[f, 'mean_position_pos'], \
                                    catalogue_list[t2].loc[f2, 'mean_position_pos'])
                    if dist <= linking_dist:
                        temp_id1 = catalogue_list[t].loc[f, 'tempID']
                        temp_id2 = catalogue_list[t2].loc[f2, 'tempID']
                        #Assign tempIDs to linked clusters
                        if np.isnan(temp_id1) and np.isnan(temp_id2):
                            catalogue_list[t].loc[f, 'tempID'] = next_temp_id
                            catalogue_list[t2].loc[f2, 'tempID'] = next_temp_id
                            next_temp_id += 1
                        elif np.isnan(temp_id1):
                            catalogue_list[t].loc[f, 'tempID'] = temp_id2
                        elif np.isnan(temp_id2):
                            catalogue_list[t2].loc[f2, 'tempID'] = temp_id1
                        elif temp_id1 != temp_id2:
                            for t3 in range(len(catalogue_list)):
                                catalogue_list[t3].loc[catalogue_list[t3]['tempID'] == temp_id2, 'tempID'] = temp_id1
    #Renaming tempID to that it goes from 1 to n
    all_tempid = get_label_list(catalogue_list, 'tempID')
    for i, tid in enumerate(np.sort(all_tempid)):
        for j in range(len(catalogue_list)):
            catalogue_list[j].loc[catalogue_list[j]['tempID'] == tid, 'tempID'] = i+1
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
            catalogue_list[i].loc[catalogue_list[i]['tempID'] == tempid_num, 'first_timestep'] = min_appearance
            catalogue_list[i].loc[catalogue_list[i]['tempID'] == tempid_num, 'last_timestep'] = max_appearance
            catalogue_list[i].loc[catalogue_list[i]['tempID'] == tempid_num, 'lifetime'] = lifetime
    return catalogue_list
