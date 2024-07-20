import os
import warnings
from pathlib import Path

from numpy import mean
from obspy import UTCDateTime as utc
from obspy import read_events
from pandas import DataFrame, Series, read_csv
from pyproj import Proj
from tqdm import tqdm

from core.Extra import (editStations, getAverageArrivals, getAverageEpicenter,
                        logger, removeOutlier, removeSparseColumns)

warnings.filterwarnings("ignore")


def createBulletin(config):
    """
    Generate formated bulletin from input catalog.

    Parameters
    ----------
    config : dict
        a dictionary contains user defined parameters.

    Returns
    -------
    None.

    """
    print("+++ Reading catalog file ...")
    rawCatalog = read_events(config["Data"]["catalogFile"])
    stations_df = read_csv(config["Data"]["stationsFile"])
    stations_df.sort_values(["code"], inplace=True)
    stations_df = editStations(stations_df, config)
    logger(f"Number of used stations: {len(stations_df)}")
    for dataType in ["train", "test"]:
        targetColumns = ["X (km)", "Y (km)", "Z (km)", "M (Ml)", "Relative Ort (s)"]
        logger(f"Working of {dataType} data")
        outDir = Path(os.path.join("inputs", dataType))
        outDir.mkdir(parents=True, exist_ok=True)
        fileOutName = os.path.join(outDir, "bulletin.csv")
        xyzm = read_csv(config["Data"][f"xyzmFile{dataType.capitalize()}"],
                        delim_whitespace=True)
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
        catalog = [rawCatalog[i] for i in eventIndices]
        xyzm.reset_index(inplace=True, drop=True)
        logger(f"Number of events in catalog and xyzm files: \
    {len(catalog)}, {len(xyzm)}")
        clat = config["StudyArea"]["lat"]
        clon = config["StudyArea"]["lon"]
        proj = Proj(
            f"+proj=sterea\
                +lon_0={clon}\
                +lat_0={clat}\
                +units=km"
        )
        maxSparsity = config["Data"]["maxSparsity"]
        columns = [code + "_P" for code in stations_df.code]
        columns += [code + "_S" for code in stations_df.code]
        columns += [f"{code}_AMP" for code in stations_df.code]
        columns += ["Avg_x", "Avg_y"]
        df = DataFrame(columns=columns)
        desc = f"+++ Processing {dataType} catalog for {len(columns)} features ... "
        for i, event in enumerate(tqdm(catalog, desc=desc, unit=" event")):
            data = {}
            preferred_origin = event.preferred_origin()
            arrivals = preferred_origin.arrivals
            amplitudes = event.amplitudes
            picks = {pick.resource_id: pick for pick in event.picks}
            for arrival in arrivals:
                arrival.update({"pick": picks[arrival.pick_id]})
            arrivals = sorted(arrivals, key=lambda x: x.pick.time)
            ref_time = utc(
                mean([arrival.pick.time.timestamp for arrival in arrivals]))
            ort = xyzm.loc[i, "ORT"]
            lat = xyzm.loc[i, "Lat"]
            lon = xyzm.loc[i, "Lon"]
            dep = xyzm.loc[i, "Dep"]
            mag = xyzm.loc[i, "Mag"]
            erh = xyzm.loc[i, "ERH"]
            erz = xyzm.loc[i, "ERZ"]
            gap = xyzm.loc[i, "GAP"]
            arrivals_P = [arv for arv in arrivals if "P" in arv.phase.upper()]
            arrivals_S = [arv for arv in arrivals if "S" in arv.phase.upper()]
            arrivals_P_mean = getAverageArrivals(arrivals_P)
            arrivals_S_mean = getAverageArrivals(arrivals_S)
            for arrivals, arrivals_mean, phase in zip(
                [arrivals_P, arrivals_S], [arrivals_P_mean, arrivals_S_mean], ["P", "S"]
            ):
                for arv in arrivals:
                    k = arv.pick.waveform_id.station_code + f"_{phase}"
                    if k in df.columns:
                        v = arv.pick.time.timestamp - arrivals_mean
                        data[k] = v
            for amplitude in amplitudes:
                k = amplitude.waveform_id.station_code + "_AMP"
                v = amplitude.generic_amplitude * 1e9
                data[k] = v
            xAvg, yAvg = getAverageEpicenter(arrivals_P, stations_df)
            data["Avg_x"] = xAvg
            data["Avg_y"] = yAvg
            new_row = data
            df_size = len(df)
            new_row["Ort"] = ref_time - utc(ort)
            new_row["Lat"] = lat
            new_row["Lon"] = lon
            new_row["Dep"] = dep
            new_row["Mag"] = mag
            if config["ExtraTargets"]["horizontal_error"]:
                new_row["ERH (km)"] = erh
                targetColumns.append("ERH (km)")
            if config["ExtraTargets"]["depth_error"]:
                new_row["ERZ (km)"] = erz
                targetColumns.append("ERZ (km)")
            if config["ExtraTargets"]["azimuthal_gap"]:
                new_row["GAP ($\\degree$)"] = gap
                targetColumns.append("GAP ($\\degree$)")
            for k, v in new_row.items():
                df.loc[df_size, k] = v
        df[["X (km)", "Y (km)"]] = df.apply(
            lambda x: Series(proj(longitude=x.Lon, latitude=x.Lat)), axis=1
        )
        df["Z (km)"] = df["Dep"]
        df["M (Ml)"] = df["Mag"]
        df["Relative Ort (s)"] = df["Ort"]
        df.drop(["Lat", "Lon", "Dep", "Mag", "Ort"], axis=1, inplace=True)
        # Delete columns containing either 75% or more than 75% NaN Values
        df = removeSparseColumns(df, dataType, maxSparsity)
        # Detect outliers and replace large values with max value after outlier removal
        df = removeOutlier(df, df.columns, dataType, "fill", False)
        df.to_csv(f"{fileOutName}", index=False, columns=df.columns)
        print(f"+++ Output file '{fileOutName}' was created successfully ...")
