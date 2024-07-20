import os

from pandas import DataFrame, read_csv
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import median_absolute_error as medae
from sklearn.metrics import r2_score as r2
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.Catalog import updateTestPredictions
from core.Extra import (decideAboutPCA, getParamDistributions, loadModel,
                        logger, r2_adj, saveModel)
from core.Report import makeReportForInputData
from time import time


def prepareTrainTest(config):
    """
    Prepare training and test files.

    Parameters
    ----------
    config : dict
        a dictionary contains user defined parameters.

    Returns
    -------
    None.

    """
    print("+++ Preparing Train and Test sets ... ")
    rndID = config["Data"]["rndID"]
    for dataType in ["train", "test"]:
        db = read_csv(os.path.join("inputs", dataType, "bulletin.csv"))
        targets_labels = ["X (km)", "Y (km)", "Z (km)", "M (Ml)", "Relative Ort (s)"]
        if config["ExtraTargets"]["horizontal_error"]:
            targets_labels.append("ERH (km)")
        if config["ExtraTargets"]["depth_error"]:
            targets_labels.append("ERZ (km)")
        if config["ExtraTargets"]["azimuthal_gap"]:
            targets_labels.append("GAP ($\\degree$)")
        y = db[targets_labels]
        X = db.drop(columns=targets_labels)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config["Data"]["testSize"] * 1e-2,
            shuffle=True,
            random_state=rndID,
        )
        for fi, name in zip(
            [X_train, X_test, y_train, y_test], ["X_train",
                                                 "X_test",
                                                 "y_train", "y_test"]
        ):
            fi.to_csv(os.path.join("inputs", dataType, f"{name}.csv"),
                      index=False, float_format="%9.3f")
            fi.index.to_frame(name="evtID").to_csv(
                os.path.join("inputs", dataType, f"{name}_id.csv"), index=False
            )
        makeReportForInputData(db, targets_labels, dataType, "bulletin")
        makeReportForInputData(X_train, targets_labels, dataType, "X_train")
        makeReportForInputData(X_test, targets_labels, dataType, "X_test")
        logger(f"Reports on {dataType} data have been created ...")


def runEstimator(config):
    """
    Run estimator.

    Parameters
    ----------
    config : dict
        a dictionary contains user defined parameters.

    Returns
    -------
    None.

    """
    print("+++ Generating Multi-Output Regressor ...")
    X_train = read_csv(os.path.join("inputs", "train", "X_train.csv"))
    y_train = read_csv(os.path.join("inputs", "train", "y_train.csv"))
    decideAboutPCA(X_train)
    logger(f"Number of features: {len(X_train.columns)}")
    logger(f"Number of targets: {len(y_train.columns)}")
    rndID = config["KFold"]["rndID"]
    nSplits = config["KFold"]["nSplits"]
    cv = KFold(nSplits, random_state=rndID, shuffle=True)
    rndID = config["Regressor"]["rndID"]
    regressor = HistGradientBoostingRegressor(
        loss=config["Regressor"]["loss"],
        random_state=rndID,
        )

    steps = [
        ("impute", KNNImputer(keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("reduce_dims", PCA()
         ) if config["DimReducer"]["dim_reduce_params"] else None,
        ("regressor", MultiOutputRegressor(regressor)),
    ]
    steps = list(filter(None, steps))
    pipe = Pipeline(steps)

    param_distributions = getParamDistributions(config)
    rndID = config["RandomizedSearchCV"]["rndID"]
    n_iter = config["RandomizedSearchCV"]["n_iter"]
    scoring = config["RandomizedSearchCV"]["scoring"]
    verbosity = config["RandomizedSearchCV"]["verbosity"]
    mor = RandomizedSearchCV(
        random_state=rndID,
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=os.cpu_count() - 2,
        verbose=verbosity,
    )
    mor.fit(X_train, y_train)
    print("+++ Multi-Output Regressor was constructed successfully ...")
    saveModel(os.path.join("results", "regressor"), mor)
    saveModel(os.path.join("results", "kfold"), cv)


def makePrediction(config):
    """
    Make prediction using test data.

    Parameters
    ----------
    config : dict
        a dictionary contains user defined parameters.

    Returns
    -------
    None.

    """
    msg = "+++ Evaluating model on Test data ..."
    logger(msg)
    print(msg)
    mor = loadModel(os.path.join("results", "regressor"))
    X_ = read_csv(os.path.join("inputs", "test", "X_test.csv"))
    y_ = read_csv(os.path.join("inputs", "test", "y_test.csv"))
    st = time()
    y_pred = mor.predict(X_)
    et = time()
    y_pred = DataFrame(y_pred, columns=y_.columns)
    y_pred.to_csv(os.path.join("results", "pred.csv"),
                  index=False, float_format="%9.3f")
    reg_sc_r2 = r2(y_, y_pred)
    reg_sc_r2_adj = r2_adj(y_, y_pred, reg_sc_r2)
    information = [
        f"Process time for predicting {len(X_)} events is: {et-st:.3f} s",
        "R2, R2-adjusted score is: {reg_sc_r2:f},\
{reg_sc_r2_adj:f}".format(reg_sc_r2=reg_sc_r2, reg_sc_r2_adj=reg_sc_r2_adj),
        "R2, MAE, RMSE, MEDAE of Latitudes is: {reg_score_r2:f}, {MAE:0.3f},\
{RMSE:0.3f}, {MEDAE:0.3f} km".format(
            reg_score_r2=r2(y_["Y (km)"], y_pred["Y (km)"]),
            MAE=mae(y_["Y (km)"], y_pred["Y (km)"]),
            RMSE=rmse(y_["Y (km)"], y_pred["Y (km)"]),
            MEDAE=medae(y_["Y (km)"], y_pred["Y (km)"]),
        ),
        "R2, MAE, RMSE, MEDAE of Longitudes is: {reg_score_r2:f}, {MAE:0.3f},\
{RMSE:0.3f}, {MEDAE:0.3f} km".format(
            reg_score_r2=r2(y_["X (km)"], y_pred["X (km)"]),
            MAE=mae(y_["X (km)"], y_pred["X (km)"]),
            RMSE=rmse(y_["X (km)"], y_pred["X (km)"]),
            MEDAE=medae(y_["X (km)"], y_pred["X (km)"]),
        ),
        "R2, MAE RMSE, MEDAE of Depths is: {reg_score_r2:f}, {MAE:0.3f},\
{RMSE:0.3f}, {MEDAE:0.3f} km".format(
            reg_score_r2=r2(y_["Z (km)"], y_pred["Z (km)"]),
            MAE=mae(y_["Z (km)"], y_pred["Z (km)"]),
            RMSE=rmse(y_["Z (km)"], y_pred["Z (km)"]),
            MEDAE=medae(y_["Z (km)"], y_pred["Z (km)"]),
        ),
        "R2, MAE, RMSE, MEDAE of Magnitudes is: {reg_score_r2:f}, {MAE:0.3f},\
{RMSE:0.3f}, {MEDAE:0.3f} ml".format(
            reg_score_r2=r2(y_["M (Ml)"], y_pred["M (Ml)"]),
            MAE=mae(y_["M (Ml)"], y_pred["M (Ml)"]),
            RMSE=rmse(y_["M (Ml)"], y_pred["M (Ml)"]),
            MEDAE=medae(y_["M (Ml)"], y_pred["M (Ml)"]),
        ),
        "R2, MAE, RMSE, MEDAE of Orts is: {reg_score_r2:f}, {MAE:0.3f},\
{RMSE:0.3f}, {MEDAE:0.3f} s".format(
            reg_score_r2=r2(y_["Relative Ort (s)"], y_pred["Relative Ort (s)"]),
            MAE=mae(y_["Relative Ort (s)"], y_pred["Relative Ort (s)"]),
            RMSE=rmse(y_["Relative Ort (s)"], y_pred["Relative Ort (s)"]),
            MEDAE=medae(y_["Relative Ort (s)"], y_pred["Relative Ort (s)"]),
        ),
        "R2, MAE, RMSE, MEDAE of ERH is: {reg_score_r2:f}, {MAE:0.3f},\
{RMSE:0.3f}, {MEDAE:0.3f} km".format(
            reg_score_r2=r2(y_["ERH (km)"], y_pred["ERH (km)"]),
            MAE=mae(y_["ERH (km)"], y_pred["ERH (km)"]),
            RMSE=rmse(y_["ERH (km)"], y_pred["ERH (km)"]),
            MEDAE=medae(y_["ERH (km)"], y_pred["ERH (km)"]),
        ) if config["ExtraTargets"]["horizontal_error"] else None,
        "R2, MAE, RMSE, MEDAE of ERZ is: {reg_score_r2:f}, {MAE:0.3f},\
{RMSE:0.3f}, {MEDAE:0.3f} km".format(
            reg_score_r2=r2(y_["ERZ (km)"], y_pred["ERZ (km)"]),
            MAE=mae(y_["ERZ (km)"], y_pred["ERZ (km)"]),
            RMSE=rmse(y_["ERZ (km)"], y_pred["ERZ (km)"]),
            MEDAE=medae(y_["ERZ (km)"], y_pred["ERZ (km)"]),
        ) if config["ExtraTargets"]["depth_error"] else None,
        "R2, MAE, RMSE, MEDAE of GAP is: {reg_score_r2:f}, {MAE:0.3f},\
{RMSE:0.3f}, {MEDAE:0.3f} deg".format(
            reg_score_r2=r2(y_["GAP ($\\degree$)"], y_pred["GAP ($\\degree$)"]),
            MAE=mae(y_["GAP ($\\degree$)"], y_pred["GAP ($\\degree$)"]),
            RMSE=rmse(y_["GAP ($\\degree$)"], y_pred["GAP ($\\degree$)"]),
            MEDAE=medae(y_["GAP ($\\degree$)"], y_pred["GAP ($\\degree$)"]),
        ) if config["ExtraTargets"]["azimuthal_gap"] else None,
    ]
    information = filter(None, information)
    for inf in information:
        print(inf)
        logger(inf)
    if config["Regressor"]["regressor_params"]:
        logger("Best Hyperparametsr:")
        logger(f"Best score is: {mor.best_score_:.2f}")
        for k, v in mor.best_params_.items():
            msg = f"{k}: {v:.3f}"
            logger(msg)
    updateTestPredictions(config)
