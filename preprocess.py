import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold

raw = pd.read_csv("data.csv")
# print raw.head()

# removing all rows containing nan in shot_made_flag column
nona = raw[pd.notnull(raw['shot_made_flag'])]

# # plot coordinates
# alpha = 0.02
# plt.figure(figsize=(10,10))

# # loc_x and loc_y	
# plt.subplot(121)
# plt.scatter(nona.loc_x, nona.loc_y, color='blue', alpha=alpha)
# plt.title("loc_x and loc_y")

# # lat and lon
# plt.subplot(122)
# plt.scatter(nona.lon, nona.lat, color='yellow', alpha=alpha)
# plt.title("lon and lat")

# plt.show()

raw['dist'] = np.sqrt(raw['loc_x']**2 + raw['loc_y']**2)

loc_x_zero = raw['loc_x'] == 0
raw['angle'] = np.array([0]*len(raw))
raw['angle'][~loc_x_zero] = np.arctan(raw['loc_y'][~loc_x_zero] / raw['loc_x'][~loc_x_zero])
raw['angle'][loc_x_zero] = np.pi / 2

raw['remaining_time'] = raw['minutes_remaining']*60 + raw['seconds_remaining']

print nona.action_type.unique()
print nona.combined_shot_type.unique()
print nona.shot_type.unique()

# plt.scatter(raw.dist, raw.shot_distance)


raw['season'] = raw['season'].apply(lambda x: int(x.split('-')[1]) )
print raw['season'].unique()

# import matplotlib.cm as cm
# plt.figure(figsize=(20,10))

# def scatter_plot_by_category(feat):
#     alpha = 0.1
#     gs = nona.groupby(feat)
#     cs = cm.rainbow(np.linspace(0, 1, len(gs)))
#     for g, c in zip(gs, cs):
#         plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)

# # shot_zone_area
# plt.subplot(131)
# scatter_plot_by_category('shot_zone_area')
# plt.title('shot_zone_area')

# # shot_zone_basic
# plt.subplot(132)
# scatter_plot_by_category('shot_zone_basic')
# plt.title('shot_zone_basic')

# # shot_zone_range
# plt.subplot(133)
# scatter_plot_by_category('shot_zone_range')
# plt.title('shot_zone_range')
# plt.show()


# Dropping unwanted variables
drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', 'matchup', 'lon', 'lat', 'seconds_remaining', 'minutes_remaining', 'shot_distance', 'loc_x', 'loc_y', 'game_event_id', 'game_id', 'game_date']
for drop in drops:
    raw = raw.drop(drop, 1)


# make dummy variables
# turn categorical variables into dummy variables
categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
for var in categorical_vars:
    raw = pd.concat([raw, pd.get_dummies(raw[var], prefix=var)], 1)
    raw = raw.drop(var, 1)

# Seperating training and testing sets
df = raw[pd.notnull(raw['shot_made_flag'])]
submission = raw[pd.isnull(raw['shot_made_flag'])]
submission = submission.drop('shot_made_flag', 1)

# Train and target sets
train = df.drop('shot_made_flag', 1) 	# training parameters
train_y = df['shot_made_flag']			# target column

df.to_csv("df.csv",index=False)
submission.to_csv("sub.csv",index=False)