################## LIBRARIES ###################
import pandas as pd
import numpy as np
import xgboost as xgb

df = pd.read_csv('df.csv')
sub = pd.read_csv('sub.csv')
train = df.drop('shot_made_flag', 1) 	# training parameters
train_y = df['shot_made_flag']			# target column

xgb_tr = xgb.DMatrix(train, label=train_y, missing=np.NaN)
xgb_ts = xgb.DMatrix(sub, missing=np.NaN)


############## Prediction model #################
param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.1
param['max_delta_step'] = 1000
param['max_depth'] = 20		#20
# param['num_class'] = 2
param['lambda']=0.01
param['subsample'] = 0.85
# param['colsample_bytree'] = 1
param['gamma'] = 1
param['min_child_weight'] = 17
num_round = 10					#2000

gbm = xgb.train(param,xgb_tr,num_round)
test_pred = gbm.predict(xgb_ts)

print test_pred

sub = pd.read_csv("sample_submission.csv")
sub['shot_made_flag'] = test_pred
sub.to_csv("subxgMD20.csv", index=False)