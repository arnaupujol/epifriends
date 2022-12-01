#This module contains utility functions to be called from epifriends

import numpy as np
import pandas as pd
import geopandas

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
            label_list = df_list[i].loc[mask, label].unique()
        else:
            label_list = np.unique(np.concatenate((label_list, df_list[i].loc[mask, label].unique())))
    return label_list


def clean_unknown_data(x, y, test = None, keep_null_tests = True, \
                       verbose = True):
    """
    This method removes all the cases with any missing value
    in either x or y.

    Parameters:
    -----------
    x: np.array, list or pd.Series
        Vector of x positions
    y: np.array, list or pd.Series
        Vector of y positions
    test: np.array, list or pd.Series
        Vector of test results (if applied)
    keep_null_tests: bool, int or float
        It defines how to treat the missing test results. If True, they are kept
        as missing, that will included foci, contributing to the total size and
        the p-value but not to the number of positives, negatives and
        positivity. If False, they are removed and not used. If int or float,
        the value is assigned to them, being interpreted as positive for 1 and
        negative for 0
    verbose: bool
        It specifies if information on the process is printed

    Returns:
    --------
    x: np.array, list or pd.Series
        Vector of x positions of elements with all x and
        y available
    y: np.array, list or pd.Series
        Vector of y positions of elements with all x and
        y available
    test: np.array, list or pd.Series
        Vector of test results (if applied) of elements
        with all x and y available

    """
    if verbose:
        print("Data with unknown positions will be excluede from the analysis")
    #Defining mask to remove samples with no geolocations available
    mask = np.isfinite(np.array(x))&np.isfinite(np.array(y))
    if test is None:
        #Redefining x, y and test without the unknown locations
        x, y = np.array(x)[mask], np.array(y)[mask]
        return x, y
    else:
        if type(keep_null_tests) in [float, int]:
            test = np.array(test, dtype = float)
            test[np.isnan(test)] = keep_null_tests
            if verbose:
                print("assigning to missing test results the value", keep_null_tests)
        elif keep_null_tests is False:
            mask = mask&np.isfinite(np.array(test, dtype = float))
            if verbose:
                print("missing test results will be removed from the analysis")
        elif keep_null_tests is True:
            pass
            if verbose:
                print("missing test results will be kept as unknown for the analysis")
        else:
            raise Exception("Error: wrong assignment of keep_null_tests")
        x, y = np.array(x)[mask], np.array(y)[mask]
        test = np.array(test, dtype = float)[mask]
        return x, y, test

def xy_from_geo(coord_df):
    """
    This method outputs the arrays of x and y coordinates
    from the geopandas geometry data.

    Parameters:
    -----------
    coord_df: geopandas.GeoDataFrame
        Geopandas dataframe with geometry variable

    Returns:
    --------
    x: np.array
        Vector of x geographical positions
    y: np.array
        Vector of y geographical positions
    """
    x = np.array(coord_df['geometry'].apply(lambda p:p.x))
    y = np.array(coord_df['geometry'].apply(lambda p:p.y))
    return x, y

def latlon2xy(lon, lat, to_epsg, verbose = True):
    """
    This method transforms the latitude and longitud
    coordinates from to cartesian coordinates in meters.

    Parameters:
    -----------
    lon: np.array
        Vector of longitude positions
    lat: np.array
        Vector of latitude positions
    to_epsg: int
        EPSG number for the projection to use
    verbose: bool
        It specifies if information on the process is printed

    Returns:
    --------
    x: np.array
        Vector of x geographical positions in meters
    y: np.array
        Vector of y geographical positions in meters
    """
    #create a geopandas dataframe with it's EPSG
    coord_df = geopandas.GeoDataFrame({'x' : lon, 'y' : lat}, \
                                      geometry = geopandas.points_from_xy(lon, lat))
    coord_df = coord_df.set_crs(epsg=4326)
    #Assing new projected coordinates
    if to_epsg is None:
        #Set as optimal projection in Mozambique
        if verbose:
            print("reprojecting coordinates to EPSG: 32736")
        coord_df = coord_df.to_crs(epsg=32736)
    else:
        if verbose:
            print("reprojecting coordinates to EPSG:", to_epsg)
        coord_df = coord_df.to_crs(epsg=to_epsg)
    x, y = xy_from_geo(coord_df)
    return x, y

def get_2dpositions(x, y, in_latlon = False, to_epsg = None, verbose = True):
    """
    This method generate a 2d-vector of cartesian positions from
    the x, y data.

    Parameters:
    -----------
    x: np.array
        Vector of x geographical positions
    y: np.array
        Vector of y geographical positions
    in_latlon: bool
        If True, x and y coordinates are treated as longitude and
        latitude respectively, otherwise they are treated as
        cartesian coordinates
    to_epsg: int
        If in_latlon is True, x and y are reprojected to this EPSG
    verbose: bool
        It specifies if information on the process is printed

    Return:
    -------
    positions: np.ndarray
        An array with the position parameters with shape (n,2),
        where n is the number of positions
    """
    if in_latlon:
        x, y = latlon2xy(x, y, to_epsg = to_epsg, verbose = verbose)
    positions = np.array((x, y)).T
    return positions
