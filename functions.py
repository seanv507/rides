import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal
import numpy as np
import xgboost as xgb
#TODO deal with categorical data
import datetime
from math import fabs

def dist(lat1, lat2, lng1, lng2):
    lat1 = lat1.radians()
    lat2 = lat2.radians()
    lng1 = lng1.radians()
    lng2 = lng2.radians()

    km_dist = (
        (
            (lat1.sin()) * (lat2.sin())
            +  (lat1.cos())
            *(lat2.cos())
            * ((lng2 - lng1).cos())
        ).arccos()
        * 6371
    )
    return km_dist

def test_dist():
    df = pl.DataFrame({'start_lat':[32], 'end_lat': [33], 'start_lng': [24], 'end_lng':[38]})
    actual = df.select(
        dist = dist(pl.col("start_lat"),pl.col("end_lat"), pl.col("start_lng"),pl.col("end_lng")),
    )
    expected = pl.Series([1316.7])
    # https://www.omnicalculator.com/other/latitude-longitude-distance
    assert fabs(actual[0,0] - expected[0]) < 0.1, f"{expected} {actual}"


def preprocess(df):
    df = df.with_columns(
        dist=dist(pl.col("start_lat"),pl.col("end_lat"), pl.col("start_lng"),pl.col("end_lng")),
        start_time=pl.col("start_time").str.to_datetime(),
    )
    return df

def split_data(df, train_cutoff_time):
    df_train,df_test = (
        df.with_columns(
            train_test = pl.when(pl.col("start_time") <= train_cutoff_time).then(pl.lit("train")).otherwise(pl.lit("test"))
        ).partition_by("train_test")
    )
    return df_train, df_test

def create_cells(df, lat_bins, lng_bins, use_quantiles: bool):
    df = cut_time(df)
    if use_quantiles:
        df, lat_bins_out, lng_bins_out = qcut_geo(df, lat_bins, lng_bins)
    else:
        df, lat_bins_out, lng_bins_out = cut_geo(df, lat_bins, lng_bins)
    df_grp = aggregate_to_cells(df)
    df_grp_all =  add_empty_cells(df_grp, ["start_lat_cut", "start_lng_cut", "start_time_cut"])
    return df_grp_all, lat_bins_out, lng_bins_out

def cut_time(df):
    df = df.with_columns(
        start_time_cut = pl.col("start_time").dt.truncate("15m"),
    )
    return df

def qcut_geo(df, bins_lat_qnt, bins_lng_qnt):
    df = df.with_columns(
        start_lat_cut = pl.col("start_lat").qcut(bins_lat_qnt),
        start_lng_cut = pl.col("start_lng").qcut(bins_lng_qnt),
    )
    lat_bins_out = get_right(df.get_column("start_lat_cut"))
    lng_bins_out = get_right(df.get_column("start_lng_cut"))
    return df, lat_bins_out, lng_bins_out

def cut_geo(df, bins_lat_qnt, bins_lng_qnt):
    df = df.with_columns(
        start_lat_cut = pl.col("start_lat").cut(bins_lat_qnt),
        start_lng_cut = pl.col("start_lng").cut(bins_lng_qnt),
    )
    lat_bins_out = get_right(df.get_column("start_lat_cut"))
    lng_bins_out = get_right(df.get_column("start_lng_cut"))
    
    return df, lat_bins_out, lng_bins_out


def get_right(bin_cat):
    vals = bin_cat.cat.get_categories().str.split(", ").list.get(1).str.strip_chars_end("]").cast(pl.Float64)
    vals =vals.filter(vals.is_finite())
    return vals

# def test_add_cuts():
#     input = pl.DataFrame(
#         {"start_time": [datetime.datetime(2024,3,1,12,17)]*6,
#          "start_lat": [23, 23, 24, 24, 26,26],
#          "start_lng": [27, 28, 29, 30, 31, 32]
#          } )
#     actual = add_cuts(input, 3,6)
#     print(actual)
#     # expected = XXXX


def aggregate_to_cells(df):
    df_grp = df.group_by(
        ["start_lat_cut", "start_lng_cut", "start_time_cut"]
    ).agg(cell_count = pl.col("dist").len(), 
          dist_sum = pl.col("dist").sum(),
          dist_mean = pl.col("dist").mean(), 
          ride_value_sum = pl.col("ride_value").sum(), 
          ride_value_mean = pl.col("ride_value").mean()
    )
    return df_grp


def test_aggregate_to_cells():
    input = pl.DataFrame({
        "start_lat_cut":[],
        "start_lng_cut": [],
        "start_time_cut": [],
    }
    )


def add_empty_cells(df_grp, index_cols):
    index_col = index_cols[0]
    levels = df_grp.select(pl.col(index_col).unique())
    for index_col in index_cols[1:]:
        level = df_grp.select(pl.col(index_col).unique())
        levels = levels.join(level, how='cross')
    df_grp = (
        df_grp
        .join(levels, on=index_cols,how="full", coalesce=True)
        .with_columns(
            cs.matches("(ride_value)|(dist)|(cell_count)").fill_null(0),)
    )

    # add missing cells (ie no rides booked)
    return df_grp


def test_add_empty_cells():
    input = pl.DataFrame(
        {"a": [1,1,2,2],
         "b": [1,1,1, 2],
         "cell_count": [1,2,3,4]}
    )
    actual = add_empty_cells(input, ["a", "b"])

    expected = pl.DataFrame(
        {"a": [1,1,1,2,2],
         "b": [1,1,2,1,2],
         "cell_count": [1,2,0, 3,4]}
    )
    print(f"test_add_empty_cells {actual}")
    assert_frame_equal(expected,actual)


def add_features(df_grp):
    df_grp = add_coord(df_grp)
    df_grp = add_time_features(df_grp)
    df_grp = add_lag_features(df_grp)
    return df_grp

def add_coord(df):
    """extract (left) value of interval for plotting etc.
    left chosen for simplicity in POC
    """
    start_lat_cut_cats = df.get_column("start_lat_cut").cat.get_categories()
    start_lng_cut_cats = df.get_column("start_lng_cut").cat.get_categories()
    df = df.with_columns(
        start_lat_flr=
            pl.col("start_lat_cut").cast(pl.String).str.split(",").list.get(0).str.strip_chars_start("(").cast(pl.Float64),
        start_lat_idx = pl.col("start_lat_cut").replace_strict(start_lat_cut_cats,list(range(len(start_lat_cut_cats)))),
        start_lng_flr= 
            df.get_column("start_lng_cut").cast(pl.String).str.split(",").list.get(0).str.strip_chars_start("(").cast(pl.Float64),
        start_lng_idx = pl.col("start_lng_cut").replace_strict(start_lng_cut_cats,list(range(len(start_lng_cut_cats)))),
    )
    return df



def add_time_features(df):
    df = df.with_columns(
        start_time_day=pl.col("start_time_cut").dt.weekday(),
        # week etc should be picked up by model.
        start_time_cut_time=pl.col("start_time_cut").dt.hour().cast(pl.Int64) * 60 + pl.col("start_time_cut").dt.minute()
    )
    return df

def test_add_time_features():
    input = pl.DataFrame({"start_time_cut": [datetime.datetime(2012,1,3,12,17), datetime.datetime(2012,1,3,14,50)]})

    expected = pl.DataFrame(
        {"start_time_cut": [datetime.datetime(2012,1,3,12,17), datetime.datetime(2012,1,3,14,50)],
         "start_time_day": [2,2],
         "start_time_cut_time": [12*60+17, 14*60+50]
         }, schema={"start_time_cut":pl.Datetime,"start_time_day":pl.Int8, "start_time_cut_time": pl.Int64})
    actual = add_time_features(input)
    print(expected, actual)
    assert_frame_equal(expected, actual,)

def add_lag_features(df):
    df_lag = (
        df
        .with_columns(
            [
                pl.col("cell_count").shift(lag).over(["start_lat_cut", "start_lng_cut"]).alias(f"cell_count_lag_{lag}")
                for lag in [1,2, 3]
            ]
        )
    )                                          
    return df_lag

def null_deviance(counts, e_count=None):
    """ score for the null model of predicting the average cell count (as baseline)"""
    if e_count is None:
        e_count = np.mean(counts)
    dev = 2* (np.where(counts, counts*np.log (counts/e_count),0) + (e_count - counts))
    return np.mean(dev)

def area(lat_cell,lng_cell):
    lat_mid = (lat_cell.left + lat_cell.right)/2
    lng_mid = (lng_cell.left + lng_cell.right)/2
    lat_dist = dist(lat_cell.left, lat_cell.right,lng_mid, lng_mid)
    lng_dist = dist(lat_mid,lat_mid,lng_cell.left, lng_cell.right)
    area = lat_dist*lng_dist
    return area


def area_vec(lat_cells, lng_cells):
    
    lat_lefts = lat_cells.apply(left).astype(float)
    lat_rights = lat_cells.apply(right).astype(float)
    lat_cells.apply(left)
    lng_lefts = lng_cells.apply(left).astype(float)
    lng_rights = lng_cells.apply(right).astype(float)
    lat_mids = (lat_lefts + lat_rights)/2
    lng_mids = (lng_lefts + lng_rights)/2
    lat_dists = dist(lat_lefts, lat_rights,lng_mids, lng_mids)
    lng_dists = dist(lat_mids,lat_mids,lng_lefts, lng_rights)
    areas = lat_dists * lng_dists
    return areas

def left(x):
    return x.left

def right(x):
    return x.right

