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
params = {
    'base_score': 0.5, 
    'colsample_bylevel': 1,
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'max_depth': 100,
    'min_child_weight': 1,
    'n_estimators': 117,
    'nthread': -1,
    'objective': 'binary:logistic',
    'seed': 27,
    # 'silent': True,
    'subsample': 0.8
}
num_round = 10

gbm = xgb.train(params, xgb_tr, num_round)
test_pred = gbm.predict(xgb_ts)

print test_pred

sub = pd.read_csv("sample_submission.csv")
sub['shot_made_flag'] = test_pred
sub.to_csv("cv_xgboost_n_est.csv", index=False)