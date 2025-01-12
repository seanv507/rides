#%%
import pandas as pd 
import polars as pl
import plotnine as p9
import numpy as np
# acos(sin(lat1)*sin(lat2)+cos(lat1)*cos(lat2)*cos(lon2-lon1))*6371 (6371 is Earth radius in km.)

def dist(lat1,lat2, lon1,lon2):
    km_dist = np.arccos(
        np.sin(np.radians(lat1))*np.sin(np.radians(lat2))
        +np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.cos(np.radians(lon2-lon1)))*6371
    return km_dist

# %%
df = pd.read_csv("data/robotex5.csv")
# %%


    # start_time - time when the order was made
    # start_lat - latitude of the order's pick-up point
    # start_lng - longitude of the order's pick-up point
    # end_lat - latitude of the order's destination point
    # end_lng - longitude of the order's destination point
    # ride_value - how much monetary value is in this particular ride

#%%
# histogram of start times

df.shape
# (627210, 6)
# %%
df.dtypes
# %%
df = (
    df
    .assign(
        dist = dist(df.start_lat,df.end_lat,df.start_lng,df.end_lng),
        start_time=pd.to_datetime(df.start_time),
        is_train = lambda df: df.start_time <="2022-03-22",
        )
)
# %%
df_train = df[df["is_train"]].copy()
# 28 days of rides

# histogram of lat/longitude

# 11:40
#15:40
#%%
# Explore the data and suggest a solution 
# to guide the drivers towards areas with
#  higher expected demand at given time and location
# Build and document a baseline model for your solution
# Describe how you would design and deploy such a model
# data exploration
# describe each variable
# calculate distance
# review distance vs value

# split into train and test set
# split lat/long 
# identify long distances lat/lng
# 
#%%
df_train = df_train.assign(
    is_whale = lambda df: df.ride_value>200
)


#%%
df_whale_watch = pd.concat((
    df_train[~df_train["is_whale"]].sample(10000),
    df_train[df_train["is_whale"]],
    )
)
df_whale_watch["id"] = np.arange(len(df_whale_watch))
plot_data = df_whale_watch.melt(id_vars=["id"],value_vars = ["start_lat", "end_lat", ])
p9.ggplot(plot_data,p9.aes())
#%%
def add_cuts(df, bins):
    df = df.assign(
        start_lat_cut = pd.qcut(df.start_lat,bins),
        start_lng_cut = pd.qcut(df.start_lng,bins),
        start_time_cut = df.start_time.dt.floor("15min")
    )
    return df
def add_time(df):
    df = df.assign(
        start_time_day = df.index.get_level_values("start_time_cut").to_series().dt.day_of_week.values,
        start_time_cut_time = df.index.get_level_values("start_time_cut").to_series().dt.time.values,
    )
    return df
#%%
df_train = add_cuts(df_train,50)

#df_train = add_time(df_train)
#%%



df_train_grp = (
    df_train.groupby(
    ["start_lat_cut", "start_lng_cut", "start_time_cut"],observed=True)
    .agg({"dist":['count','sum', 'mean'],'ride_value':["sum",'mean']})
)
#%%
# get valid values
levels = df_train_grp.index.levels
cartesian = pd.MultiIndex.from_product(levels)

# align data
df_train_grp_align = df_train_grp.reindex(cartesian,fill_value=0)
# pull out bin values
# add time variables
#844
#%%
df_train_grp_align["start_lat_flr"] = df_train_grp_align.index.get_level_values(0).to_series().apply(lambda ser: ser.left).values
df_train_grp_align["start_lng_flr"] = df_train_grp_align.index.get_level_values(1).to_series().apply(lambda ser: ser.left).values
df_train_grp_align = add_time(df_train_grp_align)
#%%
col_names = [f"{name[0]}_{name[1]}".rstrip("_") for name in df_train_grp_align.columns.values]
df_train_grp_align.columns = col_names


#%%
zz = df_train_grp_align.query("(start_lat_flr==59.416200) & (start_lng_flr==24.793000)")
zz = zz.reset_index("start_time_cut")

p9.ggplot(zz.query('start_time_cut<"2022-03-08"'), p9.aes(x="start_time_cut", y="dist_count", colour="start_time_day"))+p9.geom_line()

# %%
p9.ggplot(zz.query('start_time_cut<"2022-03-08"'), p9.aes(x="start_time_cut_time", y="dist_count", colour="start_time_day"))+p9.geom_line()
#%%

plot_data = (
    df_train_grp_align.copy()
)
col_names = [f"{name[0]}_{name[1]}".rstrip("_") for name in df_train_grp_align.columns.values]
plot_data.columns = col_names
plot_data = (
    plot_data
    .groupby(["start_lat_flr", "start_lng_flr"],observed=True,as_index=False)
    ["dist_count"].mean()
)
# todo
# split by big ride /normal and plot



#todo identify far rides
#%%
(
    p9.ggplot(
        plot_data,
        p9.aes(x="start_lat_flr",y="start_lng_flr",fill="dist_count"))+p9.geom_tile()
)
#%%
plot_data = df_train.assign(is_regular=lambda df: df.ride_value<200)
(
    p9.ggplot(plot_data,
              p9.aes(x="start_lat",=))+p9.geom_histogram(


)
    #%%
rides = (
    df_new
    .sort_values(by="ride_value",ascending=False)
    .assign(cum_sum= lambda df:df.ride_value.cumsum(),
            cum_sum_frac = lambda df:df.cum_sum/df.cum_sum.iat[-1])
)

#%%
p9.ggplot(rides.assign(index=np.arange(len(rides))).head(1000),p9.aes(x="index",y="cum_sum_frac"))+p9.geom_line()

#%%
# first 250 rides 44% of total value
# we assume these are genuine and not bad data/fraud etc (in this synthetic data set)
# one would investigate further in reality
# this could be illustrating a pareto principal that 90% of sales come fro


# %% [markdown]
# Explore the data and suggest a solution to guide the drivers towards areas with higher expected demand at given time and location

#% Build and document a baseline model for your solution
    Describe how you would design and deploy such a model
    Describe how to communicate model recommendations to drivers
    Think through and describe the design of the experiment  that would validate your solution for live operations taking into account marketplace specifics

# %%
df_new.ride_value.quantile([.99,.999,.9999])

# %%

# %%

# %%
p9.ggplot(df,p9.aes(x="start_lat"))+p9.geom_histogram()
p9.ggplot(df,p9.aes(x="end_lat"))+p9.geom_histogram()
#%%
p9.ggplot(df.sample(10000),p9.aes(x="ride_value"))+p9.geom_histogram()

#%%ride_value vs dist
plot_data = df[["start_lat", "end_lat"]].melt()
#p9.ggplot(df,p9.aes(x="start_lat"))+p9.geom_histogram()
# %%
p9.ggplot(plot_data.sample(10000),p9.aes(x='value',colour='variable'))+p9.geom_histogram()
# %%
(
    p9.ggplot(df_new,p9.aes(x="dist",y="ride_value")) + 
    p9.geom_point(alpha=0.3) +
    p9.geom_smooth(method="lm",colour='red')
)
# %%
