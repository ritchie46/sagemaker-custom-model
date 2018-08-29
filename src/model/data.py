import pandas as pd
import numpy as np


def prepare_column(series, minimal=None, maximal=None, log=False):
    """
    :param series: (Pandas Series)
    :param minimal: (flt) Truncate a minimal value.
    :param maximal: (flt) Truncate a maximal value.
    :param log:  (bool) Take the log of the array.
    :return: (Series)
    """
    series = series.copy()
    if minimal:
        series[series < minimal] = minimal
    if maximal:
        series[series > maximal] = maximal
    if log:
        series = np.log(series)
    return series


def to_dummies(series, delete=None, remove_redundant=True, prefix=None):
    """

    :param series: (Pandas Series)
    :param delete: (list) Containing the column names that should be deleted.
    :param remove_redundant: (bool)
    :return: (Series)
    """
    df = pd.get_dummies(series, prefix)
    if delete:
        df = df.drop(delete, axis=1)
    if remove_redundant:
        df = df.iloc[:, :-1]
    return df


def prepare_data(df, col_select, scaler, mean_fill):
    """
    :param df: (DataFrame)
    :param col_select: (list) Columns to select from DataFrame
    :param scaler: (sklearn Scaler)
    :param mean_fill: (array) replace nans with these mean values. (These are required when using trained model)
    :returns: (array) features
    """
    df = df.copy()

    if 'fysieke_capaciteit' in df.columns:
        df.loc[df['fysieke_capaciteit'] == 'nxan', 'fysieke_capaciteit'] = 'nan'
    if 'g_woz' in df.columns:
        df['g_woz'].loc[df['g_woz'] == 0] = df['g_woz'].mean()

    # Fill missing values with mean
    data = pd.concat((df[['p_koopw', 'p_soz_wb', 'p_soz_ww']],
                  df[['aantal_koffieshopsgemeente', 'g_woz']],
                  df[['sjv_delta_factor', 'sjv_laag_delta_factor']].sum(1).to_frame('sjv_delta_factor'), # heavily correlated, sum them
                  prepare_column(df['inhoud'], maximal=12e3),
                  prepare_column(df['oppervlakteverblijfsobject'], maximal=20e3, minimal=0.001, log=True),
                  df[['sme_powerdown_count',
        'sme_powerdown_duur_mean', 'luci_leegstand_duur']],
                  ), axis=1)

    data = data.fillna(mean_fill)

    # Fill missing values with -1. Another option would be to make an extra column indicating that the value is missing.
    data = pd.concat((data,
                      df['spkl_count'],
                      df['pti_geweigerd'],
                      df[['sme_a_strong_magnetic_dc_field_has_been_detected',
                          'sme_fraud_attempt_for_mbus_1', 'sme_fraud_attempt_for_mbus_2',
                          'sme_intrusion_detection_user_tried_to_gain_access_with_a_wrong_password',
                          #        'sme_the_meter_cover_has_been_removed', # highly correlated with terminal cover that has been removed.
                          'sme_the_terminal_cover_has_been_removed']]
                      ), axis=1)
    data = data.fillna(-1)
    columns = list(data.columns)

    data = pd.DataFrame(scaler.transform(data), columns=columns)

    dummies = pd.concat((
        to_dummies(df['huureigendom'], prefix='huureigendom'),
        to_dummies(df['pandstatus'], prefix='pandstatus'),
        to_dummies(df['verbruiksegment'], prefix='verbruiksegment'),
        to_dummies(df['fysieke_capaciteit'], prefix='capaciteit'),
        #                       to_dummies(df[['profielcategorie']], prefix='categorie'),
        to_dummies(df['verblijfsobjectgebruiksdoel']),
        #                       to_dummies(df['mos_label'], prefix='mos')
    ), axis=1).fillna(0).reset_index()
    data = pd.concat((dummies, data), axis=1)

    columns = columns + list(dummies.columns)
    # In case some events have not occurred in the excerpt of the data, some columns will be missing.
    diff = set(col_select) - set(columns)
    for col in diff:
        data[col] = 0

    return data[col_select].values
