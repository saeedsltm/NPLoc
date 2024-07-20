import os

from pandas import DataFrame, read_csv

from core.Extra import loadModel
from time import time


def predict(config):
    """
    Make prediction on test data.

    Parameters
    ----------
    config : dict
        a dictionary contains user defined parameters.

    Returns
    -------
    None.

    """
    print("+++ Preparing Data ... ")
    db = read_csv(os.path.join("inputs", "test", "bulletin.csv"))
    outputFile = os.path.join("results", "bulletin_test_relocated.csv")
    targets_labels = ["X (km)", "Y (km)", "Z (km)", "Relative Ort (s)", "M (Ml)"]
    if config["ExtraTargets"]["horizontal_error"]:
        targets_labels.append("ERH (km)")
    if config["ExtraTargets"]["depth_error"]:
        targets_labels.append("ERZ (km)")
    if config["ExtraTargets"]["azimuthal_gap"]:
        targets_labels.append("GAP ($\\degree$)")
    y = db[targets_labels]
    X = db.drop(columns=targets_labels)
    mor = loadModel(os.path.join("results", "regressor"))
    st = time()
    y_pred = mor.predict(X)
    et = time()
    y_pred = DataFrame(y_pred, columns=y.columns)
    with open(outputFile, "w") as f:
        y_pred.to_string(f, columns=targets_labels, index=False, formatters={
            "X (km)": "{:7.3f}".format,
            "Y (km)": "{:7.3f}".format,
            "Z (km)": "{:7.3f}".format,
            "Relative Ort (s)": "{:5.3f}".format,
            "M (Ml)": "{:4.1f}".format,
            "ERH (km)": "{:5.2f}".format,
            "ERZ (km)": "{:5.2f}".format,
            "GAP ($\\degree$)": "{:5.2f}".format,
        })
    print(f"+++ Prediction time for {len(y_pred)} events is: {et-st:.3f} s")
