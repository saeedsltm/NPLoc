import os
import sys
import time
from pathlib import Path
from shutil import copy

import skops.io as sio
from numpy import (abs, array, average, mean, nan, nanpercentile, round_, sqrt,
                   sum, tril_indices, where)
from obspy.geodetics.base import degrees2kilometers as d2k
from pandas import DataFrame, Series
from pyproj import Proj
from scipy.stats import loguniform
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from yaml import SafeLoader, load


def logger(message, mode="a"):
    """
    Function for generating logs.

    Parameters
    ----------
    message : str
        message to be loged.
    mode : str, optional
        loging mode. The default is "a".

    Returns
    -------
    None.

    """
    logPath = os.path.join("results", "running.log")
    message = time.strftime("%d %b %Y %H:%M:%S - ") + message + "\n"
    with open(logPath, mode) as f:
        f.write(message)


def readConfiguration():
    """
    Read configuration file.

    Returns
    -------
    config : dict
        a dictionary contains user defined parameters.

    """
    createResultsFolder()
    if not os.path.exists("config.yml"):
        msg = "+++ Could not find configuration file! Aborting ..."
        print(msg)
        sys.exit()
    with open("config.yml") as f:
        config = load(f, Loader=SafeLoader)
    msg = "+++ Configuration file was loaded successfully ..."
    print(msg)
    copy("config.yml", os.path.join("results", "config.yml"))
    return config


def createResultsFolder():
    """
    Create directory for saving results.

    Returns
    -------
    None.

    """
    resultsPath = Path("results")
    inputsPath = Path("inputs")
    resultsPath.mkdir(parents=True, exist_ok=True)
    inputsPath.mkdir(parents=True, exist_ok=True)


def editStations(stations_df, config):
    """
    Edit station dataframe for removing stains beyond critical distance.

    Parameters
    ----------
    stations_df : pandas.DataFrame
        station dataframe.
    config : dict
        a dictionary contains user defined parameters.

    Returns
    -------
    stations_df : pandas.DataFrame
        edited station dataframe.

    """
    stationsFilePath = os.path.join("inputs", "stations.csv")
    clat = config["StudyArea"]["lat"]
    clon = config["StudyArea"]["lon"]
    R = config["StudyArea"]["maxRadius"]
    proj = Proj(f"+proj=sterea\
            +lon_0={clon}\
            +lat_0={clat}\
            +units=km")
    stations_df = stations_df[["code", "lat", "lon", "elv"]]
    stations_df[["x", "y"]] = stations_df.apply(
        lambda x: Series(
            proj(longitude=x.lon, latitude=x.lat)), axis=1)
    stations_df["z"] = stations_df["elv"]
    stations_df[["r"]] = stations_df.apply(
        lambda x: Series(sqrt(x.x**2 + x.y**2)), axis=1)
    a = len(stations_df)
    stations_df = stations_df[stations_df["r"] <= R]
    b = a - len(stations_df)
    logger(f"Number of stations excluded due to critical radius < {R:0.1f} km: {b}",
           "w")
    stations_df.to_csv(stationsFilePath, index=False, float_format="%8.3f")
    return stations_df


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


def saveModel(fileName, model):
    """
    Save objects using skops library.

    Parameters
    ----------
    fileName : str
        output file name.
    model : object
        model to be saved.

    Returns
    -------
    None.

    """
    sio.dump(model, "{0}.sko".format(fileName))
    print(f"+++ {model.__class__.__name__} model has been saved in {fileName} ...")


def loadModel(fileName):
    """
    Load objects using skops library.

    Parameters
    ----------
    fileName : str
        input file name.

    Returns
    -------
    model : object
        model to be saved.

    """
    model = sio.load("{0}.sko".format(fileName), trusted=True)
    print(f"+++ {fileName.split(os.sep)[-1]} model has been loaded ...")
    return model


def scaleData(data, columns):
    """

    Parameters
    ----------
    data : DataFrame
        a pandas dataframe contains data of events..

    Returns
    -------
    data_scaled : DataFrame
        a standard-Scaled pandas dataframe contains data of events.

    """
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    db = DataFrame(data_scaled, columns=columns)
    saveModel(os.path.join("results", "scaler"), scaler)
    return db


def r2_adj(y_test, y_pred, r2):
    """
    Adjusted version of R-Squared.

    Parameters
    ----------
    y_test : pandas.DataFrame
        actual targets.
    y_pred : pandas.DataFrame
        predted targets.
    r2 : object
        calculated R-Squared.

    Returns
    -------
    object
        adjusted version of R-Squared..

    """
    n = y_test.shape[0]
    p = y_test.shape[1]
    return 1-(1-r2)*(n-1)/(n-p-1)


def getParamDistributions(config):
    """
    Get hyperparameters of estimator for tunning.

    Parameters
    ----------
    config : dict
        a dictionary contains user defined parametrs.

    Returns
    -------
    param_distributions : dict
        a dictionary contains hyperparameters of estimator.

    """
    param_distributions = {}
    if config["DimReducer"]["dim_reduce_params"]:
        items = config["DimReducer"]["dim_reduce_params"].items()
        param_distributions.update({"reduce_dims__{k:s}".format(k=key): eval(
            value) for key, value in items})
        logger("PCA is enabled")
    else:
        logger("PCA is not enabled")
    if config["Regressor"]["regressor_params"]:
        items = config["Regressor"]["regressor_params"].items()
        param_distributions.update(
            {f"regressor__estimator__{key:s}": eval(value) for key, value in items})
        logger("Regressor hyperparameters testing is enabled for the followings:")
        for k, v in param_distributions.items():
            logger(f"{k}: {v.__class__.__name__}")
    else:
        logger("Regressor hyperparameters testing is not enabled")
    return param_distributions


def removeOutlier(df, columns, dataType, method="fill", silent=True):
    """
    Remove outliers from dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe contains features.
    columns : list
        a list of features lable.
    method : str, optional
        method to manipulate outliers. The default is "fill".

    Returns
    -------
    df : pandas.DataFrame
        a cleaned dataframe contains features.

    """
    q1 = nanpercentile(df[columns], 25, axis=0)
    q3 = nanpercentile(df[columns], 75, axis=0)
    IQR = q3 - q1
    lower_limit = q1 - (1.5*IQR)
    upper_limit = q3 + (1.5*IQR)
    mask = df[columns] < lower_limit
    mask = df[columns] > upper_limit

    col_means = df.mean(axis=0, skipna=True)
    df = DataFrame(where(df[columns] < lower_limit,
                   col_means, df), columns=df.columns)
    df = DataFrame(where(df[columns] > upper_limit,
                   col_means, df), columns=df.columns)

    outliersSize = len(mask.count())
    outliersPerc = 1e2*len(mask.count())/df.size
    if not silent:
        logger(f"Number of removed/filled outliers for {dataType} data: \
{outliersSize} ~ {outliersPerc:.1f}%")
    # df[columns] = df[columns].mask(mask)
    # if method == "fill":
    #     df[columns] = df[columns].mask(mask).fillna(df[columns].mean())
    # elif method == "drop":
    #     df.dropna(inplace=True)
    return df


def removeSparseColumns(df, dataType, max_sparsity=75.0):
    """
    Remove sparse columns from a given dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        input dataframe.
    max_sparsity : float, optional
        maximum sparisity. The default is 75.0.

    Returns
    -------
    df : pandas.DataFrame
        a cleaned dataframe.

    """
    a = len(df.columns)
    min_count = int(((100-max_sparsity)/100)*df.shape[0] + 1)
    df.dropna(axis=1, thresh=min_count, inplace=True)
    b = a - len(df.columns)
    logger(f"Number of removed fetures due to sparsity for {dataType} data: \
{b} ~ {100*b/a:0.2f}%")
    return df


def getAverageArrivals(arrivals):
    """
    Calculate weighted average arrivals.

    Parameters
    ----------
    arrivals : obspy.arrivals
        arrival times.

    Returns
    -------
    float
        weighted average arrival time.

    """
    if not len(arrivals):
        return nan
    times = [arr.pick.time.timestamp for arr in arrivals]
    times = array(times).reshape(-1, 1)
    dt = [arr.pick.time.timestamp -
          arrivals[0].pick.time.timestamp for arr in arrivals[::-1]]
    dt = array(dt).reshape(-1, 1)
    scaler = MinMaxScaler()
    weights = scaler.fit_transform(dt)
    try:
        arrAvg = average(times, weights=weights)
    except ZeroDivisionError:
        arrAvg = times
    return arrAvg


def getAverageEpicenter(arrivals, stations_df):
    """
    Calculate weighted average distances.

    Parameters
    ----------
    arrivals : obspy.arrivals
        arrival times.
    stations_df : pandas.DataFrame
        a dataframe contains station information.

    Returns
    -------
    x : float
        weighted average of station longitude.
    y : float
        weighted average of station latitude.

    """
    codes = [arr.pick.waveform_id.station_code for arr in arrivals]
    times = [arr.pick.time.timestamp -
             arrivals[0].pick.time.timestamp for arr in arrivals[::-1]]
    times = array(times).reshape(-1, 1)
    scaler = MinMaxScaler()
    weights = scaler.fit_transform(times)
    x = array([stations_df[stations_df.code == code].x.values[0] for code in codes])
    y = array([stations_df[stations_df.code == code].y.values[0] for code in codes])
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    x = average(x, weights=weights)
    y = average(y, weights=weights)
    return x, y


def decideAboutPCA(X_train):
    """
    Make decision if PCS is needed.

    Parameters
    ----------
    X_train : pandas.DataFrame
        a dataframe contains features.

    Returns
    -------
    None.

    """
    corr = X_train.corr()
    corr = abs(corr.values[tril_indices(corr.shape[0], -1)] > 0.3)
    corr_percent_30 = 1e2*sum(corr)/len(corr)
    if corr_percent_30 < 50.0:
        logger(f"{corr_percent_30:0.1f}% of total input features have correlations >\
0.3, then applying PCA is not recommended")
    elif corr_percent_30 >= 50.0 and corr_percent_30 < 70.0:
        logger(f"{corr_percent_30:0.1f}% of total input features have correlations >\
0.3, then applying PCA slightly recommended")
    elif corr_percent_30 >= 70.0:
        logger(f"{corr_percent_30:0.1f}% of total input features have correlations >\
0.3, then applying PCA highly recommended")


def handleNone(value, degree=False, dtype="float"):
    """Handle missing values

    Args:
        value (float): a float value
        degree (bool, optional): whether convert to degree or not.
        Defaults to False.

    Returns:
        float: handled value
    """
    if value is None:
        return nan
    else:
        if degree:
            return d2k(value)
        return int(value) if dtype == "int" else value


def getHer(event):
    """Get horizontal error of event

    Args:
        event (obspy.event): an obspy event

    Returns:
        float: event horizontal error
    """
    if event.origins[0].latitude_errors.uncertainty:
        x = event.origins[0].latitude_errors.uncertainty
        y = event.origins[0].longitude_errors.uncertainty
        return round(d2k(sqrt(x**2 + y**2)), 1)
    else:
        return None


def getZer(event):
    """Get depth error of event

    Args:
        event (obspy.event): an obspy event

    Returns:
        float: event depth error
    """
    if event.origins[0].depth_errors.uncertainty:
        return event.origins[0].depth_errors.uncertainty*0.001
    else:
        return None


def getRMS(arrivals):
    """
    Get RMS of the event

    Parameters
    ----------
    arrivals : obspy.arrivals
        list of arrivals.

    Returns
    -------
    weighted_rms : float
        weighted time residuals.

    """
    time_residuals = array([
        arrival.time_residual for arrival in arrivals if isinstance(
            arrival.time_residual, float)
    ])
    time_weights = array([
        arrival.time_weight for arrival in arrivals if isinstance(
            arrival.time_weight, float)
    ])
    if time_residuals.size:
        weighted_rms = sum(time_weights * time_residuals **
                           2) / sum(time_weights)
        weighted_rms = sqrt(weighted_rms)
    else:
        weighted_rms = nan
    return weighted_rms


def catalog2xyzm(obsCatalog, outName):
    """Convert an Obspy catalog to xyzm file format

    Args:
        hypInp (str): file name of NORDIC file
        catalogFileName (str): file name of xyzm.dat file
    """
    outputFile = os.path.join("results", f"xyzm_{outName:s}.dat")
    catDict = {}
    for i, event in enumerate(obsCatalog):
        po = event.preferred_origin()
        preferred_magnitude = event.preferred_magnitude()
        arrivals = po.arrivals
        ort = po.time
        lat = po.latitude
        lon = po.longitude
        mag = preferred_magnitude.mag if preferred_magnitude else nan
        try:
            dep = po.depth*0.001
        except TypeError:
            dep = nan
        try:
            nus = handleNone(
                po.quality.used_station_count, dtype="int")
        except AttributeError:
            nus = nan
        nuP = len(
            [arrival.phase for arrival in arrivals if "P" in arrival.phase.upper()])
        nuS = len(
            [arrival.phase for arrival in arrivals if "S" in arrival.phase.upper()])
        mds = handleNone(
            min([handleNone(arrival.distance) for arrival in po.arrivals]),
            degree=True)
        ads = round_(handleNone(
            mean([handleNone(arrival.distance) for arrival in po.arrivals]),
            degree=True), 2)
        try:
            gap = handleNone(
                po.quality.azimuthal_gap, dtype="int")
        except AttributeError:
            gap = nan
        rms = getRMS(po.arrivals)
        erh = getHer(event)
        erz = getZer(event)
        catDict[i] = {
            "ORT": ort,
            "Lon": lon,
            "Lat": lat,
            "Dep": dep,
            "Mag": mag,
            "Nus": nus,
            "NuP": nuP,
            "NuS": nuS,
            "ADS": ads,
            "MDS": mds,
            "GAP": gap,
            "RMS": rms,
            "ERH": erh,
            "ERZ": erz,
        }
    df = DataFrame(catDict).T
    df = df.replace({"None": nan})
    with open(outputFile, "w") as f:
        df.to_string(f, index=False, formatters={
            "ORT": "{:}".format,
            "Lon": "{:7.3f}".format,
            "Lat": "{:7.3f}".format,
            "Dep": "{:7.3f}".format,
            "Mag": "{:4.1f}".format,
            "Nus": "{:3.0f}".format,
            "NuP": "{:3.0f}".format,
            "NuS": "{:3.0f}".format,
            "ADS": "{:5.1f}".format,
            "MDS": "{:5.1f}".format,
            "GAP": "{:3.0f}".format,
            "RMS": "{:5.2f}".format,
            "ERH": "{:5.1f}".format,
            "ERZ": "{:5.1f}".format,
        })
