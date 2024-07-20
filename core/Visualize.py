import os
from pathlib import Path

import proplot as plt
from numpy import (abs, arange, array, linspace, median, mgrid, nan, nanmax,
                   sqrt, tril_indices)
from obspy import read_events
from pandas import DataFrame, read_csv
from scipy.interpolate import griddata
from scipy.stats import median_abs_deviation as mad
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import learning_curve, validation_curve

from core.Extra import loadModel, removeOutlier


def createResultsFolder():
    figuresPath = Path("results")
    figuresPath.mkdir(parents=True, exist_ok=True)
    return figuresPath


def plotPCA(nComponents, pca_scores, best_nComponents):
    axesShape = array([
        [1]
    ])
    config = plt.Configurator()
    config.reset()
    plt.rc.update(
        {'fontsize': 5, 'legend.fontsize': 5,
         'label.weight': 'bold', 'title.weight': 'bold'})
    fig, axs = plt.subplots(axesShape, share=True, span=True)
    axs.format(xlocator=("maxn", 5),
               xlabel="Number of components/features",
               ylabel="CV Scores")
    [ax.grid(ls=":") for ax in axs]
    axs[0].plot(range(nComponents), pca_scores, marker="o", ms=2,
                mfc="r", mec="k", mew=0.25, c="k", ls=":", lw=0.75)
    axs[0].axvspan(best_nComponents, nComponents,
                   edgecolor="gray", facecolor="gray", alpha=0.25)
    axs[0].text(best_nComponents, y=max(pca_scores)+5, s=f"$N={best_nComponents}$")
    axs.figure.tight_layout()
    figuresPath = Path("results")
    figuresPath.mkdir(parents=True, exist_ok=True)
    figureName = os.path.join(figuresPath, "pca.png")
    fig.save(figureName)


def plotPredictionVsTest(config):
    figuresPath = createResultsFolder()
    mor = loadModel(os.path.join("results", "regressor"))
    X_test = read_csv(os.path.join("inputs", "test", "X_test.csv"))
    y_test = read_csv(os.path.join("inputs", "test", "y_test.csv"))
    y_pred = mor.predict(X_test)
    y_pred = DataFrame(y_pred, columns=y_test.columns)
    histBounds = {
        'X (km)': config["Figures"]["histBounds"]["Lon"],
        'Y (km)': config["Figures"]["histBounds"]["Lat"],
        'Z (km)': config["Figures"]["histBounds"]["Dep"],
        'M (Ml)': config["Figures"]["histBounds"]["Mag"],
        'Relative Ort (s)': config["Figures"]["histBounds"]["ORT"],
        'ERH (km)': config["Figures"]["histBounds"]["ERH"],
        'ERZ (km)': config["Figures"]["histBounds"]["ERZ"],
        'GAP ($\\degree$)': config["Figures"]["histBounds"]["GAP"]}
    for i, column in enumerate(y_test.columns):
        figureName = os.path.join(figuresPath, f"predictedVsTest_{column}.png")
        config = plt.Configurator()
        config.reset()
        plt.rc.update(
            {'fontsize': 9, 'legend.fontsize': 5, 'label.weight': 'bold'})
        axesShape = [
            [1]
        ]
        fig, axs = plt.subplots(axesShape)
        [ax.grid(ls=":") for ax in axs]
        ax = axs[0]
        x = y_test[column].values
        y = y_pred[column].values
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xymin = min(xmin, ymin)
        xymax = min(xmax, ymax)
        ax.scatter(x, y, s=5, marker="o", c="r",
                   lw=0.4, edgecolors="k", alpha=.5)
        ax.plot([0, 1], [0, 1], transform=ax.transAxes,
                ls="--", c="red")
        ax.format(title=f"{column}",
                  urtitle=f"{len(y_test)} events",
                  xlabel="Test",
                  ylabel="Predicted",
                  xlim=(xymin, xymax),
                  ylim=(xymin, xymax),
                  xlocator=("maxn", 5),
                  ylocator=("maxn", 5))
        ix = ax.inset([0.65, 0.10, 0.3, 0.3], transform="axes", zoom=False)
        ix.grid(ls=":")
        ix.spines["right"].set_visible(False)
        ix.spines["top"].set_visible(False)
        data = x - y
        M = data.mean()
        D = median(data)
        Std = data.std()
        Mad = mad(data)
        Mae = mean_absolute_error(x, y)
        Mdae = median_absolute_error(x, y)
        ix.format(
            fontsize=8,
            xlocator=("maxn", 3),
            ylocator=("maxn", 4),
            xlim=(-histBounds[column], histBounds[column]))
        ix.hist(
            data, linspace(-histBounds[column], histBounds[column], 15),
            lw=0.3, histtype="bar", filled=True,
            alpha=0.7, edgecolor="w", color="gray")
        title = '\n'.join((
            f"$\\mu_{{m}}={M:.2f}$",
            f"$\\sigma_{{m}}={Std:.2f}$",
            f"$\\mathcal{{M_{{m}}}}={Mae:.2f}$",
            f"$\\mu_{{d}}={D:.2f}$",
            f"$\\sigma_{{d}}={Mad:.2f}$",
            f"$\\mathcal{{M_{{d}}}}={Mdae:.2f}$"
        ))
        ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=6, va="top")
        fig.save(figureName)


def plotDataStatistics(config):
    """

    Parameters
    ----------
    db : DataFrame
        visulize some statistics obtained via data.

    Returns
    -------
    None.

    """
    print("+++ plotting data statistics ...")
    db_path = os.path.join("inputs", "bulletin.csv")
    db = read_csv(db_path)
    targets_labels = ["X (km)", "Y (km)", "Z (km)",
                      "M (Ml)", "Relative Ort (s)"]
    if config["ExtraTargets"]["horizontal_error"]:
        targets_labels.append("ERH (km)")
    if config["ExtraTargets"]["depth_error"]:
        targets_labels.append("ERZ (km)")
    if config["ExtraTargets"]["azimuthal_gap"]:
        targets_labels.append("GAP ($\\degree$)")
    data_df = db.drop(targets_labels, axis=1)
    df = DataFrame()
    for column in data_df.columns:
        df[column] = [data_df[column].count()]
    axesShape = array([[
        1,
    ]])
    config = plt.Configurator()
    config.reset()
    plt.rc.update(
        {'fontsize': 7, 'legend.fontsize': 5, 'label.weight': 'bold'})
    fig, axs = plt.subplots(axesShape, share=False)
    axs[0].format(
        xlabel="Features",
        xlocator=("maxn", 5),
        ylabel="Samples",
        ylocator=("maxn", 5),
        xrotation=90,
        xticklabelsize=5)
    [ax.grid(ls=":") for ax in axs]
    axs[0].bar(df.T, c="gray", ec="k")
    figuresPath = Path("results")
    figuresPath.mkdir(parents=True, exist_ok=True)
    figureName = os.path.join(
        figuresPath, "statistics.png")
    fig.save(figureName)

    # Covariance matrix plot
    for key in ["_P", "_S", "_AMP"]:
        axesShape = array([[
            1,
        ]])
        config = plt.Configurator()
        config.reset()
        plt.rc.update(
            {'fontsize': 7, 'legend.fontsize': 5, 'label.weight': 'bold'})
        fig, axs = plt.subplots(axesShape, share=False)
        data = data_df[[code for code in data_df.columns if key in code]]
        corr = data.corr()
        corr.values[tril_indices(corr.shape[0], -1)] = nan
        axs[0].heatmap(corr, cmap="ColdHot", vmin=-1, vmax=1, N=100, lw=0.5, ec="k",
                       labels=True, precision=2,
                       labels_kw={"weight": "bold", "size": "xx-small"},
                       clip_on=False)
        axs[0].format(
            title=f"{key.split('_')[-1]}-phase correlation coefficients",
            xloc="top", yloc="right",
            yreverse=True, ticklabelweight="bold",
            alpha=0, linewidth=0, tickpad=4,
            xrotation=90
        )
        figuresPath = Path("results")
        figuresPath.mkdir(parents=True, exist_ok=True)
        figureName = os.path.join(
            figuresPath, f"Correlations{key}.png")
        fig.save(figureName)


def validationCurve(
        regressor,
        X, y,
        param_name,
        param_range,
        cv,
        xaxisLog=False):
    train_scores, valid_scores = validation_curve(
        regressor,
        X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        n_jobs=os.cpu_count() - 2)
    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    valid_scores_mean = valid_scores.mean(axis=1)
    valid_scores_std = valid_scores.std(axis=1)
    axesShape = array([[
        1,
    ]])
    config = plt.Configurator()
    config.reset()
    plt.rc.update(
        {'fontsize': 7, 'legend.fontsize': 5, 'label.weight': 'bold'})
    colors = ["orange7", "lime7", "teal7", "violet7"]
    fig, axs = plt.subplots(axesShape, share=False)
    axs[0].format(
        xlabel=f"Hyperparameter: {param_name.split('__')[-1]}",
        xlocator=("maxn", 5),
        ylabel="Score",
        ylim=(0.0, 1.0),
        ylocator=("maxn", 5))
    [ax.grid(ls=":") for ax in axs]
    if xaxisLog:
        axs[0].semilogx(
            param_range, train_scores_mean,
            label="Training score", color=colors[0],
            lw=0.5, ls="-.", legend="lr", legend_kw={"ncol": 1})
    else:
        axs[0].plot(
            param_range, train_scores_mean,
            label="Training score", color=colors[0],
            lw=0.5, ls="-.", legend="lr", legend_kw={"ncol": 1})
    axs[0].fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2, color=colors[0], lw=0.5)
    if xaxisLog:
        axs[0].semilogx(
            param_range, valid_scores_mean,
            label="CV score",
            color=colors[1], lw=0.5, legend="lr", legend_kw={"ncol": 1})
    else:
        axs[0].plot(
            param_range, valid_scores_mean,
            label="CV score",
            color=colors[1], lw=0.5, legend="lr", legend_kw={"ncol": 1})
    axs[0].fill_between(
        param_range,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.2, color=colors[1], lw=0.5)
    figuresPath = Path("results")
    figuresPath.mkdir(parents=True, exist_ok=True)
    figureName = os.path.join(
        figuresPath, f"validationCurve_{param_name.split('__')[-1]}.png")
    fig.save(figureName)


def plotValidationCurve(config):
    mor = loadModel(os.path.join("results", "regressor"))
    cv = loadModel(os.path.join("results", "kfold"))
    X_train = read_csv(os.path.join("inputs", "X_train.csv"))
    y_train = read_csv(os.path.join("inputs", "y_train.csv"))
    logs = ["regressor__estimator__learning_rate",
            "regressor__estimator__l2_regularization"]
    for key, _ in mor.param_distributions.items():
        print("+++ Plotting Valdiation Curve for:", key)
        validationCurve(
            mor.best_estimator_,
            X_train, y_train,
            key,
            sorted(mor.cv_results_[f"param_{key}"].data),
            cv,
            xaxisLog=True if key in logs else False)


def learningCurve(
        regressor,
        X, y,
        train_sizes,
        scoring,
        cv):
    train_sizes, train_scores, valid_scores, fit_times, _ = learning_curve(
        regressor,
        X, y,
        scoring=scoring,
        cv=cv,
        train_sizes=train_sizes,
        return_times=True)
    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    valid_scores_mean = valid_scores.mean(axis=1)
    valid_scores_std = valid_scores.std(axis=1)
    fit_times_mean = fit_times.mean(axis=1)
    fit_times_std = fit_times.std(axis=1)
    axesShape = array([
        [1, 2, 3]
    ])
    config = plt.Configurator()
    config.reset()
    fig, axs = plt.subplots(axesShape, share=False)
    axs[0].format(
        xlabel="Training samples",
        ylabel="Score",
        ylim=(0.0, 1.0),
        xlocator=("maxn", 5),
        ylocator=("maxn", 5))
    [ax.grid(ls=":") for ax in axs]

    axs[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="darkorange"
    )
    axs[0].fill_between(
        train_sizes,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.1,
        color="navy"
    )
    axs[0].plot(
        train_sizes, train_scores_mean, m="o", ls=":", ms=3, lw=1,
        color="darkorange", label="Training score",
        legend="lr", legend_kw={"ncol": 1})
    axs[0].plot(
        train_sizes, valid_scores_mean, m="o", ls=":", ms=3, lw=1, color="navy",
        label="Cross-validation score",
        legend="lr", legend_kw={"ncol": 1})

    axs[1].format(
        xlabel="Training samples",
        ylabel="Fit times (s)")
    axs[1].plot(train_sizes, fit_times_mean, m="o", ls=":", ms=3, lw=1, color="navy")
    axs[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1)

    axs[2].format(
        xlabel="Fit times (s)",
        ylabel="Score")
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    valid_scores_mean_sorted = valid_scores_mean[fit_time_argsort]
    valid_scores_std_sorted = valid_scores_std[fit_time_argsort]
    axs[2].plot(fit_time_sorted, valid_scores_mean_sorted, m="o", ls=":", ms=3, lw=1,
                color="navy")
    axs[2].fill_between(
        fit_time_sorted,
        valid_scores_mean_sorted - valid_scores_std_sorted,
        valid_scores_mean_sorted + valid_scores_std_sorted,
        alpha=0.1)

    figuresPath = Path("results")
    figuresPath.mkdir(parents=True, exist_ok=True)
    figureName = os.path.join(figuresPath, "learningCurve")
    fig.save(figureName)


def plotLearningCurve(config):
    mor = loadModel(os.path.join("results", "regressor"))
    cv = loadModel(os.path.join("results", "kfold"))
    X_train = read_csv(os.path.join("inputs", "X_train.csv"))
    y_train = read_csv(os.path.join("inputs", "y_train.csv"))
    scoring = config["Scoring"]["metric"]
    train_sizes = linspace(0.1, 1.0, 5)
    learningCurve(mor, X_train, y_train, train_sizes, scoring, cv)


def plotPartialDependence():
    mor = loadModel(os.path.join("results", "regressor"))
    X_train = read_csv(os.path.join("inputs", "X_test.csv"))
    y_train = read_csv(os.path.join("inputs", "y_test.csv"))
    features = X_train.columns
    targets = y_train.columns
    for t, target in enumerate(targets):
        name = "{0}".format(target)
        axesShape = array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10]
        ])
        config = plt.Configurator()
        config.reset()
        fig, axs = plt.subplots(axesShape, sharey=True, sharex=False)
        axs.format(
            ylabel="Partial dependence - {0}".format(target), ylocator=('maxn', 5))
        [ax.grid(ls=":") for ax in axs]
        _ = PartialDependenceDisplay.from_estimator(
            mor, X_train, features, ax=axs, target=t)
        figuresPath = Path("results")
        figuresPath.mkdir(parents=True, exist_ok=True)
        figureName = os.path.join(figuresPath, "pdp_{0}.png".format(name))
        fig.save(figureName)


def plotPermutationImportance():
    mor = loadModel(os.path.join("results", "regressor"))
    X_train = read_csv(os.path.join("inputs", "X_train.csv"))
    y_train = read_csv(os.path.join("inputs", "y_train.csv"))
    result = permutation_importance(
        mor,
        X_train, y_train,
        n_repeats=10,
        random_state=0,
        n_jobs=os.cpu_count()
    )
    sorted_importances_idx = result.importances_mean.argsort()
    importances = DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X_train.columns[sorted_importances_idx],
    )
    axesShape = array([
        [1]
    ])
    config = plt.Configurator()
    config.reset()
    plt.rc.update(
        {'fontsize': 5, 'legend.fontsize': 5, 'label.weight': 'bold'})
    fig, axs = plt.subplots(axesShape)
    axs.format(title="Permutation Importances",
               xlocator=('maxn', 5),
               xlabel="Decrease in accuracy score",
               ylabel="Features",
               yticklabelsize=5)
    [ax.grid(ls=":") for ax in axs]
    axs[0].boxh(importances, means=True, marker="", meancolor="r", fillcolor="gray4")
    axs[0].axvline(x=0, color="k", linestyle="--")
    axs[0].figure.tight_layout()
    figuresPath = Path("results")
    figuresPath.mkdir(parents=True, exist_ok=True)
    figureName = os.path.join(figuresPath, "permutationImportances.png")
    fig.save(figureName)


def plotMisfit(config, y_, y_pred, stations_df, target):
    print(f"+++ Plotting {target} using Test data ...")
    misfit_df = DataFrame()
    target_labels = ["X (km)", "Y (km)", "Z (km)", "M (Ml)", "Relative Ort (s)"]
    target_names = ["dx", "dy", "dz", "dm", "dt"]
    if config["ExtraTargets"]["horizontal_error"]:
        target_labels.append("ERH (km)")
        target_names.append("derh")
    if config["ExtraTargets"]["depth_error"]:
        target_labels.append("ERZ (km)")
        target_names.append("derz")
    if config["ExtraTargets"]["azimuthal_gap"]:
        target_labels.append("GAP ($\\degree$)")
        target_names.append("dgap")

    for target_name, target_label in zip(target_names, target_labels):
        misfit_df[target_label] = y_[target_label] - y_pred[target_label]
        exec(
            f"{target_name}=removeOutlier(misfit_df, [target_label], 'test')[target_label]")
    if target == "Horizontal misfit":
        misfit = sqrt(eval("dx")**2 + eval("dy")**2)
        unit = "(km)"
        errorMax = config["Figures"]["misfit"]["errorMaxHorizontal"]
    elif target == "Depth misfit":
        misfit = abs(eval("dz"))
        unit = "(km)"
        errorMax = config["Figures"]["misfit"]["errorMaxDepth"]
    elif target == "Magnitude misfit":
        misfit = abs(eval("dm"))
        unit = "(Ml)"
        errorMax = config["Figures"]["misfit"]["errorMaxMagnitude"]
    elif target == "Origin-Time misfit":
        misfit = abs(eval("dt"))
        unit = "(s)"
        errorMax = config["Figures"]["misfit"]["errorMaxOriginTime"]
    elif target == "Horizontal-Error misfit":
        misfit = abs(eval("derh"))
        unit = "(km)"
        errorMax = config["Figures"]["misfit"]["errorMaxERH"]
    elif target == "Depth-Error misfit":
        misfit = abs(eval("derz"))
        unit = "(km)"
        errorMax = config["Figures"]["misfit"]["errorMaxERZ"]
    elif target == "Gap misfit":
        misfit = abs(eval("dgap"))
        unit = "$(\\degree)$"
        errorMax = config["Figures"]["misfit"]["errorMaxGap"]
    x = y_["X (km)"]
    y = y_["Y (km)"]
    xMin, xMax = config["Figures"]["misfit"]["xLim"]
    yMin, yMax = config["Figures"]["misfit"]["yLim"]
    xGridPts = complex(config["Figures"]["misfit"]["xGridPts"])
    yGridPts = complex(config["Figures"]["misfit"]["yGridPts"])
    axesShape = array([
        [1]
    ])
    config = plt.Configurator()
    config.reset()
    plt.rc.update(
        {'fontsize': 5, 'legend.fontsize': 5,
         'label.weight': 'bold', 'title.weight': 'bold'})
    fig, axs = plt.subplots(axesShape, share=True, span=True)
    axs.format(xlocator=("maxn", 5),
               xlabel="Easting (km)",
               ylabel="Northing (km)",
               xlim=(xMin, xMax),
               ylim=(yMin, yMax))
    [ax.grid(True, ls=":") for ax in axs]

    axs[0].format(ctitle=f"{target} {unit}", urtitle=f"{len(y_)} events")
    axs[0].scatter(stations_df.x, stations_df.y, marker="^", lw=0.75,
                   edgecolors="k", s=20, c="w", zorder=10, alpha=0.3)
    for _, row in stations_df.iterrows():
        adj_y = 0.04*(yMax - yMin)
        axs[0].text(x=row.x, y=row.y-adj_y,
                    s=row.code, border=True, borderinvert=True, borderwidth=1,
                    **{"weight": "bold", "size": "x-small", "ha": "center"},
                    alpha=0.3, clip_on=True)
    X, Y = mgrid[xMin:xMax:xGridPts, yMin:yMax:yGridPts]
    T = griddata((x, y), misfit, (X, Y), method="linear", rescale=True)
    # errorMax = min([errorMax, nanmax(T)])
    levels = linspace(0, errorMax, 20)
    cs = axs[0].contourf(X, Y, T, levels=levels, extend="both",
                         cmap="YlOrRd", vmax=errorMax)
    axs[0].contour(X, Y, T, levels=levels, colors="k", lw=.2, alpha=0.2)
    axs[0].set_facecolor("gray")
    axs[0].scatter(x, y, c='k', s=1, alpha=0.1, marker='.')
    axs[0].colorbar(
        cs, loc="lr", label=f"Misfit {unit}", ticks=linspace(0, errorMax, 11),
        ticklabelsize=4, ticklen=2)
    axs.figure.tight_layout()
    figuresPath = Path("results")
    figuresPath.mkdir(parents=True, exist_ok=True)
    figureName = os.path.join(figuresPath, f"{target}_test.png")
    fig.save(figureName)


def plotMisfits(config):
    mor = loadModel(os.path.join("results", "regressor"))
    stationsFilePath = os.path.join("inputs", "stations.csv")
    stations_df = read_csv(stationsFilePath)
    X_ = read_csv(os.path.join("inputs", "test", "X_test.csv"))
    y_ = read_csv(os.path.join("inputs", "test", "y_test.csv"))
    y_pred = mor.predict(X_)
    y_pred = DataFrame(y_pred, columns=y_.columns)
    plotMisfit(config, y_, y_pred, stations_df,
               "Horizontal misfit")
    plotMisfit(config, y_, y_pred, stations_df,
               "Depth misfit")
    plotMisfit(config, y_, y_pred, stations_df,
               "Magnitude misfit")
    plotMisfit(config, y_, y_pred, stations_df,
               "Origin-Time misfit")
    if config["ExtraTargets"]["horizontal_error"]:
        plotMisfit(config, y_, y_pred, stations_df,
                   "Horizontal-Error misfit")
    if config["ExtraTargets"]["depth_error"]:
        plotMisfit(config, y_, y_pred, stations_df,
                   "Depth-Error misfit")
    if config["ExtraTargets"]["azimuthal_gap"]:
        plotMisfit(config, y_, y_pred, stations_df,
                   "Gap misfit")


def plotMAE(config):
    mor = loadModel(os.path.join("results", "regressor"))
    xyzmUpdated = read_csv(os.path.join(
        "results", "xyzm_upatedCatalog.dat"), delim_whitespace=True)
    X_ = read_csv(os.path.join("inputs", "test", "X_test.csv"))
    y_ = read_csv(os.path.join("inputs", "test", "y_test.csv"))
    X_ids = read_csv(os.path.join("inputs", "test", "X_test_id.csv"))
    xyzmUpdated["eventID"] = X_ids.evtID
    y_pred = mor.predict(X_)
    y_pred = DataFrame(y_pred, columns=y_.columns)
    diff = abs(y_pred - y_)

    axesShape = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])
    pltConfig = plt.Configurator()
    pltConfig.reset()
    plt.rc.update(
        {'fontsize': 8, 'legend.fontsize': 5,
         'label.weight': 'bold', 'title.weight': 'bold'})
    fig, axs = plt.subplots(axesShape, share=False)
    axs.format(ylocator=("maxn", 5),
               xformatter="%.0f",
               abc="A)",
               abcloc="ul"
               )
    [ax.grid(True, ls=":") for ax in axs]

    # axs-01
    X, Y = [], []
    for NuP in sorted(xyzmUpdated.NuP.unique()):
        axs[0].format(xlabel="Number of P phases",
                      ylabel="|Misfit on X (km)|",
                      ylim=config["Figures"]["maeMisfitBounds"]["Lon"]
                      )
        x = NuP
        y = diff["X (km)"][xyzmUpdated.NuP == NuP].median()
        X.append(x)
        Y.append(y)
        axs[0].plot(x, y, m="o", mfc="gray", mec="k", mew=0.5)
        num_points = len(diff["X (km)"][xyzmUpdated.NuP == NuP].values)
        axs[0].annotate(f"{num_points}", (x, y),
                        xytext=(1, 4), textcoords="offset points")
    axs[0].plot(X, Y, m="", ls=":", lw=1.5, color="gray", dashes=(1, 0.5))
    ox = axs[0].altx(label="Number of event", color="olive", zorder=1)
    bins = linspace(*config["Figures"]["maeMisfitBounds"]["Lon"], 16)
    ox.hist(diff["X (km)"].values, bins=bins, orientation="horizontal",
            lw=0.75, histtype="bar", filled=True, edgecolor="k", color="pale olive",
            alpha=0.7)
    axs[0].set_zorder(ox.get_zorder() + 1)

    # axs-02
    X, Y = [], []
    for NuP in sorted(xyzmUpdated.NuP.unique()):
        axs[1].format(xlabel="Number of P phases",
                      ylabel="|Misfit on Y (km)|",
                      ylim=config["Figures"]["maeMisfitBounds"]["Lat"]
                      )
        x = NuP
        y = diff["Y (km)"][xyzmUpdated.NuP == NuP].median()
        X.append(x)
        Y.append(y)
        axs[1].plot(x, y, m="o", mfc="gray", mec="k", mew=0.5)
        num_points = len(diff["Y (km)"][xyzmUpdated.NuP == NuP].values)
        axs[1].annotate(f"{num_points}", (x, y),
                        xytext=(1, 4), textcoords="offset points")
    axs[1].plot(X, Y, m="", ls=":", lw=1.5, color="gray", dashes=(1, 0.5))
    ox = axs[1].altx(label="Number of event", color="olive", zorder=1)
    bins = linspace(*config["Figures"]["maeMisfitBounds"]["Lat"], 16)
    ox.hist(diff["Y (km)"].values, bins=bins, orientation="horizontal",
            lw=0.75, histtype="bar", filled=True, edgecolor="k", color="pale olive",
            alpha=0.7)
    axs[1].set_zorder(ox.get_zorder() + 1)

    # axs-03
    X, Y = [], []
    for NuS in sorted(xyzmUpdated.NuS.unique()):
        axs[2].format(xlabel="Number of S phases",
                      ylabel="|Misfit on Z (km)|",
                      ylim=config["Figures"]["maeMisfitBounds"]["Dep"]
                      )
        x = NuS
        y = diff["Z (km)"][xyzmUpdated.NuS == NuS].median()
        X.append(x)
        Y.append(y)
        axs[2].plot(x, y, m="o", mfc="gray", mec="k", mew=0.5)
        num_points = len(diff["Z (km)"][xyzmUpdated.NuS == NuS].values)
        axs[2].annotate(f"{num_points}", (x, y),
                        xytext=(1, 4), textcoords="offset points")
    axs[2].plot(X, Y, m="", ls=":", lw=1.5, color="gray", dashes=(1, 0.5))
    ox = axs[2].altx(label="Number of event", color="olive", zorder=1)
    bins = linspace(*config["Figures"]["maeMisfitBounds"]["Dep"], 16)
    ox.hist(diff["Z (km)"].values, bins=bins, orientation="horizontal",
            lw=0.75, histtype="bar", filled=True, edgecolor="k", color="pale olive",
            alpha=0.7)
    axs[2].set_zorder(ox.get_zorder() + 1)

    # axs-04
    X, Y = [], []
    for Nus in sorted(xyzmUpdated.Nus.unique()):
        axs[3].format(xlabel="Number of station",
                      ylabel="|Misfit on Relative Ort (s)|",
                      ylim=config["Figures"]["maeMisfitBounds"]["ORT"]
                      )
        x = Nus
        y = diff["Relative Ort (s)"][xyzmUpdated.Nus == Nus].median()
        X.append(x)
        Y.append(y)
        axs[3].plot(x, y, m="o", mfc="gray", mec="k", mew=0.5)
        num_points = len(diff["Relative Ort (s)"][xyzmUpdated.Nus == Nus].values)
        axs[3].annotate(f"{num_points}", (x, y),
                        xytext=(1, 4), textcoords="offset points")
    axs[3].plot(X, Y, m="", ls=":", lw=1.5, color="gray", dashes=(1, 0.5))
    ox = axs[3].altx(label="Number of event", color="olive", zorder=1)
    bins = linspace(*config["Figures"]["maeMisfitBounds"]["ORT"], 16)
    ox.hist(diff["Relative Ort (s)"].values, bins=bins, orientation="horizontal",
            lw=0.75, histtype="bar", filled=True, edgecolor="k", color="pale olive",
            alpha=0.7)
    axs[3].set_zorder(ox.get_zorder() + 1)


    # axs-05
    axs[4].format(ylabel="Average distance (km)",
                  xlabel="Magnitude (km)",
                  )
    xMin = xyzmUpdated.Mag.min()
    xMax = xyzmUpdated.Mag.max()
    yMin = xyzmUpdated.ADS.min()
    yMax = xyzmUpdated.ADS.max()
    xGridPts = 0.01
    yGridPts = 2.0
    x = xyzmUpdated.Mag.values
    y = xyzmUpdated.ADS.values
    misfit = diff["M (Ml)"].values
    X, Y = mgrid[xMin:xMax:xGridPts, yMin:yMax:yGridPts]
    T = griddata((x, y), misfit, (X, Y), method="linear")
    levels = linspace(0, config["Figures"]["maeMisfitBounds"]["Mag"], 10)
    cs = axs[4].contourf(X, Y, T, levels=levels, extend="both",
                         cmap="gray_r", vmax=2)
    axs[4].contour(X, Y, T, levels=levels, colors="k", lw=.2, alpha=0.2)
    axs[4].colorbar(
        cs, loc="lr", title="|Misfit on M (Ml)|",
        ticks=levels, frame=False, length=10, ticklabelsize=5, labelsize=6)

    # axs-06
    if config["ExtraTargets"]["horizontal_error"]:
        X, Y = [], []
        for Nus in sorted(xyzmUpdated.Nus.unique()):
            axs[5].format(xlabel="Number of station",
                          ylabel="|Misfit on ERH (km)|",
                          ylim=config["Figures"]["maeMisfitBounds"]["ERH"]
                          )
            x = Nus
            y = diff["ERH (km)"][xyzmUpdated.Nus == Nus].median()
            X.append(x)
            Y.append(y)
            axs[5].plot(x, y, m="o", mfc="gray", mec="k", mew=0.5)
            num_points = len(diff["ERH (km)"][xyzmUpdated.Nus == Nus].values)
            axs[5].annotate(f"{num_points}", (x, y),
                            xytext=(1, 4), textcoords="offset points")
        axs[5].plot(X, Y, m="", ls=":", lw=1.5, color="gray", dashes=(1, 0.5))
    ox = axs[5].altx(label="Number of event", color="olive", zorder=1)
    bins = linspace(*config["Figures"]["maeMisfitBounds"]["ERH"], 16)
    ox.hist(diff["ERH (km)"].values, bins=bins, orientation="horizontal",
            lw=0.75, histtype="bar", filled=True, edgecolor="k", color="pale olive",
            alpha=0.7)
    axs[5].set_zorder(ox.get_zorder() + 1)

    # axs-07
    if config["ExtraTargets"]["depth_error"]:
        X, Y = [], []
        for Nus in sorted(xyzmUpdated.Nus.unique()):
            axs[6].format(xlabel="Number of station",
                          ylabel="|Misfit on ERZ (km)|",
                          ylim=config["Figures"]["maeMisfitBounds"]["ERZ"]
                          )
            x = Nus
            y = diff["ERZ (km)"][xyzmUpdated.Nus == Nus].median()
            X.append(x)
            Y.append(y)
            axs[6].plot(x, y, m="o", mfc="gray", mec="k", mew=0.5)
            num_points = len(diff["ERZ (km)"][xyzmUpdated.Nus == Nus].values)
            axs[6].annotate(f"{num_points}", (x, y),
                            xytext=(1, 4), textcoords="offset points")
        axs[6].plot(X, Y, m="", ls=":", lw=1.5, color="gray", dashes=(1, 0.5))
        ox = axs[6].altx(label="Number of event", color="olive", zorder=1)
        bins = linspace(*config["Figures"]["maeMisfitBounds"]["ERZ"], 16)
        ox.hist(diff["ERZ (km)"].values, bins=bins, orientation="horizontal",
                lw=0.75, histtype="bar", filled=True, edgecolor="k", color="pale olive",
                alpha=0.7)
        axs[6].set_zorder(ox.get_zorder() + 1)

    # axs-08
    if config["ExtraTargets"]["azimuthal_gap"]:
        X, Y = [], []
        for Nus in sorted(xyzmUpdated.Nus.unique()):
            axs[7].format(xlabel="Number of station",
                          ylabel="|Misfit on GAP ($\\degree$)|",
                          ylim=config["Figures"]["maeMisfitBounds"]["GAP"]
                          )
            x = Nus
            y = diff["GAP ($\\degree$)"][xyzmUpdated.Nus == Nus].median()
            X.append(x)
            Y.append(y)
            axs[7].plot(x, y, m="o", mfc="gray", mec="k", mew=0.5)
            num_points = len(diff["GAP ($\\degree$)"]
                             [xyzmUpdated.Nus == Nus].values)
            axs[7].annotate(f"{num_points}", (x, y),
                            xytext=(1, 4), textcoords="offset points")
        axs[7].plot(X, Y, m="", ls=":", lw=1.5, color="gray", dashes=(1, 0.5))
    ox = axs[7].altx(label="Number of event", color="olive", zorder=1)
    bins = linspace(*config["Figures"]["maeMisfitBounds"]["GAP"], 16)
    ox.hist(diff["GAP ($\\degree$)"].values, bins=bins, orientation="horizontal",
            lw=0.75, histtype="bar", filled=True, edgecolor="k", color="pale olive",
            alpha=0.7)
    axs[7].set_zorder(ox.get_zorder() + 1)

    figuresPath = Path("results")
    figuresPath.mkdir(parents=True, exist_ok=True)
    figureName = os.path.join(figuresPath, "target_MAE.png")
    fig.save(figureName)
