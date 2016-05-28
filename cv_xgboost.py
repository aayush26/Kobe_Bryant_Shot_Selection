################## LIBRARIES ###################
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
from sklearn.cross_validation import cross_val_score, train_test_split


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train, label=train_y, missing=np.NaN)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds, metrics=['logloss'])
        alg.set_params(n_estimators=cvresult.shape[0])
   	
   	print cvresult 
    # Test params
    X_train, X_valid, y_train, y_valid = train_test_split(dtrain, predictors, test_size=0.2)
    alg.fit(X_train, y_train, eval_metric='logloss')
    y_pred = alg.predict_proba(X_valid)[:,1]
    
    result = log_loss(y_valid, y_pred)
    print(result)
    # return result


################ LOAD DATA #######################
df = pd.read_csv('df.csv')
sub = pd.read_csv('sub.csv')
train = df.drop('shot_made_flag', 1) 	# training parameters
train_y = df['shot_made_flag']			# target column


############ XGBOOST MODEL ########################
xgb_tr = xgb.DMatrix(train, label=train_y, missing=np.NaN)
xgb_ts = xgb.DMatrix(sub, missing=np.NaN)


params = {
    'base_score': 0.5, 
    'colsample_bylevel': 1,
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'max_depth': 32,
    'min_child_weight': 1,
    'n_estimators': 117,
    'nthread': -1,
    'objective': 'binary:logistic',
    'seed': 27,
    # 'silent': True,
    'subsample': 0.8,
}

clf = xgb.XGBClassifier()
clf.set_params(**params)
modelfit(clf, train, train_y)