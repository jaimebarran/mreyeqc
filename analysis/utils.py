import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

def load_and_format_data(dataset, first_iqm, move_iqms=False):
    """Load and format a dataset"""
    df = pd.read_csv(dataset)

    cols = df.columns.tolist()
    xy_index = cols.index(first_iqm)
    if move_iqms:
        df = df[cols[:xy_index] + cols[-14:] + cols[xy_index:-14]]
    # types = {col: float if "nan" not in col else bool for col in dataframe.columns}
    # dataframe = dataframe.astype(types)
    xy_index = df.columns.tolist().index(first_iqm)

    train_x = df[df.columns[xy_index:]].copy()
    train_y = df[df.columns[:xy_index]].copy()
    # Cast all columns with nan as nan
    cols_nan = train_x.columns[train_x.columns.str.contains("nan")]
    train_x[cols_nan] = train_x[cols_nan].astype(bool).astype(float)

    remove_idx = train_x.isnull().any(axis=1)
    train_x = train_x[~remove_idx]
    train_y = train_y[~remove_idx]

    remove_idx = (train_x == np.inf).any(axis=1)
    train_x = train_x[~remove_idx]
    train_y = train_y[~remove_idx]
    train_y["site_rec"] = train_y["site"] + "_" + train_y["rec"]

    # Discard feta site
    idx_feta = train_y["site"] == "feta"
    feta_x = train_x[idx_feta]
    feta_y = train_y[idx_feta]
    train_x = train_x[~idx_feta]
    train_y = train_y[~idx_feta]

    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    feta_x = feta_x.reset_index(drop=True)
    feta_y = feta_y.reset_index(drop=True)
    
    # Drop constant columns
    train_x = train_x.loc[:, train_x.var() != 0]
    return train_x, train_y, feta_x, feta_y




def pick_scaler(scaler):
    from fetal_brain_qc.qc_evaluation.preprocess import (
        GroupRobustScaler,
        GroupStandardScaler,
        PassThroughScaler,
    )

    if scaler.lower() in ("passthrough", "passthroughscaler"):
        return PassThroughScaler()
    elif scaler.lower() == "groupstandardscaler":
        return GroupStandardScaler(groupby="group")
    elif scaler.lower() == "grouprobustscaler":
        return GroupRobustScaler(groupby="group")
    elif scaler.lower() == "standardscaler":
        return StandardScaler()
    elif scaler.lower() == "robustscaler":
        return RobustScaler()
    else:
        raise ValueError



def add_group(df_ref, group):
    """Returns the `group` column from `df_ref` depending
    on the choice.
    """
    if group == "vx_size":
        return df_ref["vx_size"] > 3.0
    elif group in [
        "sub_ses",
        "sub",
        "rec",
        "site_field",
        "site",
        "model",
        "site_scanner",
        "site_rec",
    ]:
        return df_ref[group]
    elif group is None or group.lower() == "none":
        return [None for i in range(df_ref.shape[0])]
    else:
        raise NotImplementedError