import os

from pandas import DataFrame
from scipy.stats import median_abs_deviation as mad


def makeReportForInputData(df, target_coloumns, dataType, outname):
    """
    Make report on final results.

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe contains eighter train or test data.
    target_coloumns : list
        a list contains features/target.
    dataType : str
        Data type Train or Test.
    outname : str
        output nameof the report file.

    Returns
    -------
    None.

    """
    print(f"+++ Making report on {outname} ...")
    results = {}
    data = df.copy()
    for feature_name in sorted(data.columns):
        feature_length = df[feature_name].size
        feature_nans = df[feature_name].isna().sum()
        feature_mean = df[feature_name].mean()
        feature_std = df[feature_name].std()
        feature_med = df[feature_name].median()
        feature_mad = mad(df[feature_name], nan_policy="omit")
        results[feature_name] = {
            "Samples_#": feature_length,
            "NaN_#": feature_nans,
            "NaN_%": 1e2*(feature_nans/feature_length),
            "Mean": feature_mean,
            "STD": feature_std,
            "Median": feature_med,
            "MAD": feature_mad,
        }
    df = DataFrame(results)
    df = df.T.sort_values("NaN_#")
    formatters = {'Samples_#': '{:,.0f}'.format,
                  'NaN_#': '{:,.0f}'.format,
                  'NaN_%': '{:,.1f}%'.format,
                  'Mean': '{:,.3f}'.format,
                  'STD': '{:,.3f}'.format,
                  'Median': '{:,.3f}'.format,
                  'MAD': '{:,.3f}'.format}
    with open(os.path.join("inputs", dataType, f"report_{outname}.dat"), "w") as f:
        df.to_string(f, formatters=formatters)
