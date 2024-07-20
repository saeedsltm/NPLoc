import os

from numpy import diff, mean
from obspy import UTCDateTime as utc
from obspy import read_events
from obspy.core.event import Catalog
from obspy.geodetics.base import gps2dist_azimuth as gps
from obspy.geodetics.base import locations2degrees as l2d
from pandas import Series, read_csv
from pyproj import Proj
from tqdm import tqdm

from core.Extra import catalog2xyzm


def updateArrivals(evLon, evLat, arrivals, stations_df):
    """
    Update arrivals for extra propertices of distance and azimuth.

    Parameters
    ----------
    evLon : float
        event longitude.
    evLat : float
        event latitude.
    arrivals : obspy.arrivals
        a list of arrivals.
    stations_df : pandas.DataFrame
        a dataframe contains stations information.

    Returns
    -------
    arrivals :  obspy.arrivals
        a list of updated arrivals.

    """
    codes = list(set([arv.pick.waveform_id.station_code for arv in arrivals]))
    stations_df = stations_df[stations_df.code.isin(codes)]
    stations_df[["Dist"]] = stations_df.apply(
        lambda x: Series(l2d(evLat, evLon, x.lat, x.lon)), axis=1)
    stations_df[["Azim"]] = stations_df.apply(
        lambda x: Series(gps(evLat, evLon, x.lat, x.lon)[1]), axis=1)
    for arrival in arrivals:
        code = arrival.pick.waveform_id.station_code
        arrival.azimuth = stations_df[stations_df.code == code].Azim
        arrival.distance = stations_df[stations_df.code == code].Dist
    return arrivals


def computeExtraInfo(evLon, evLat, arrivals, station_df):
    """
    Calculate extra information for arrivals.

    Parameters
    ----------
    evLon : float
        event longitude.
    evLat : float
        event latitude.
    arrivals : obspy.arrivals
        a list of arrivals.
    stations_df : pandas.DataFrame
        a dataframe contains stations information.

    Returns
    -------
    Nus : int
        number of used stations.
    NuP : int
        number of P phases.
    NuS : int
        number of S phases.
    ADS : float
        average distance between event-stations.
    MDS : float
        minimum distance between event-stations.
    GAP : float
        aimuthal gap of the event.

    """
    codes = list(set([arv.pick.waveform_id.station_code for arv in arrivals]))
    station_df = station_df[station_df.code.isin(codes)]
    station_df[["Dist"]] = station_df.apply(
        lambda x: Series(gps(evLat, evLon, x.lat, x.lon)[0]*1e-3), axis=1)
    station_df[["Azim"]] = station_df.apply(
        lambda x: Series(gps(evLat, evLon, x.lat, x.lon)[1]), axis=1)
    azimuths = station_df["Azim"].sort_values()
    Dist = station_df["Dist"]
    GAP = int(max(diff(azimuths)))
    ADS = Dist.mean()
    MDS = Dist.min()
    Nus = len(codes)
    NuP = sum([1 for arrival in arrivals if arrival.phase.upper().startswith("P")])
    NuS = sum([1 for arrival in arrivals if arrival.phase.upper().startswith("S")])
    return Nus, NuP, NuS, ADS, MDS, GAP


def updateTestPredictions(config):
    """
    Update predictions for extra information.

    Parameters
    ----------
    config : dict
        a dictionary contains user defined parameters.

    Returns
    -------
    None.

    """
    events_id = read_csv(os.path.join("inputs", "test", "X_test_id.csv"))
    predictions = read_csv(os.path.join("results", "pred.csv"))
    predictions["evtID"] = events_id.values
    catalog = read_events(config["Data"]["catalogFile"])
    stationsFilePath = os.path.join("inputs", "stations.csv")
    stations_df = read_csv(stationsFilePath)
    xyzm = read_csv(config["Data"]["xyzmFileTest"], delim_whitespace=True)
    xyzm = xyzm[
        (xyzm.ORT.notna())
        & (xyzm.Lat.notna())
        & (xyzm.Lon.notna())
        & (xyzm.Dep.notna())
        & (xyzm.Mag.notna())
        & (xyzm.ERH.notna() if config["ExtraTargets"]["horizontal_error"] else True)
        & (xyzm.ERZ.notna() if config["ExtraTargets"]["depth_error"] else True)
        & (xyzm.GAP.notna() if config["ExtraTargets"]["azimuthal_gap"] else True)
    ]
    eventIndices = xyzm.index.values
    catalog = [catalog[i] for i in eventIndices]
    catalogUpdated = Catalog()
    catalogTest = Catalog()
    clat = config["StudyArea"]["lat"]
    clon = config["StudyArea"]["lon"]
    proj = Proj(f"+proj=sterea\
            +lon_0={clon}\
            +lat_0={clat}\
            +units=km")
    predictions[["Lon", "Lat"]] = predictions.apply(
        lambda x: Series(
            proj(longitude=x["X (km)"], latitude=x["Y (km)"], inverse=True)), axis=1)
    predictions["Dep"] = predictions["Z (km)"]
    predictions["Mag"] = predictions["M (Ml)"]

    desc = "+++ Reading & Updating catalog ..."
    for _, pred in tqdm(predictions.iterrows(), desc=desc, unit=" event"):
        event = catalog[int(pred.evtID)]
        catalogTest.append(event.copy())
        preferred_origin = event.preferred_origin()
        preferred_magnitude = event.preferred_magnitude()
        arrivals = preferred_origin.arrivals
        picks = {pick.resource_id: pick for pick in event.picks}
        for arrival in arrivals:
            arrival.update({"pick": picks[arrival.pick_id]})
        arrivals = sorted(arrivals, key=lambda x: x.pick.time)
        ref_time = mean(
            [arrival.pick.time.timestamp for arrival in arrivals])
        ort = utc(ref_time - pred["Relative Ort (s)"])
        lat = pred.Lat
        lon = pred.Lon
        dep = pred.Dep*1e+3
        preferred_origin.time = ort
        preferred_origin.latitude = lat
        preferred_origin.longitude = lon
        preferred_origin.depth = dep
        preferred_magnitude.mag = pred.Mag
        arrivals = updateArrivals(lon, lat, arrivals, stations_df)
        _, _, _, _, _, GAP = computeExtraInfo(lon, lat, arrivals, stations_df)
        preferred_origin.quality.azimuthal_gap = GAP
        catalogUpdated.append(event)
    print("+++ Writting updated catalog ...")
    catalogUpdated.write(os.path.join(
        "results", "upatedCatalog.out"), format="NORDIC", high_accuracy=False)
    print("+++ Writting Test catalog ...")
    catalogTest.write(os.path.join(
        "results", "testCatalog.out"), format="NORDIC", high_accuracy=False)
    catalog2xyzm(catalogUpdated, "upatedCatalog")
