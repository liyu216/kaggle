import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

train = pd.read_csv('train1.csv')
train[['pred_carbon_monoxide', 'pred_benzen', 'pred_nitrogen_oxides']] = 0, 0, 0
test = pd.read_csv('test1.csv')
ss = pd.read_csv('sample_submission.csv')


def SCORE_FUNCTION(preds, truth):
    return np.mean(np.square(np.log1p(preds) - np.log1p(truth)))


model = LGBMRegressor()
tss = TimeSeriesSplit()
FEATURES = ['deg_C', 'relative_humidity', 'absolute_humidity', 'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
            'sensor_5', 'month', 'week', 'is_weekend', 'hour'
]

scores = []
for train_idx, test_idx in tss.split(train):
    score = []
    for TARGET in ['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides']:
        predictor = '_'.join(['pred'] + TARGET.split('_')[1:])

        model.fit(train[FEATURES].iloc[train_idx], train[[TARGET]].iloc[train_idx])
        preds = np.clip(model.predict(train[FEATURES].iloc[test_idx]), a_min=0., a_max=1e100)
        score.append(SCORE_FUNCTION(preds, train[TARGET].iloc[test_idx].values))

    score.append(np.mean(score))
    scores.append(pd.DataFrame(score).T)
    print(f'{score[0] :.4f}, {score[1] :.4f}, {score[2] :.4f}, {score[3] :.4f}')

# Predictions
for TARGET in ['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides']:
    model.fit(train[FEATURES], train[[TARGET]])
    ss[TARGET] = np.clip(model.predict(test[FEATURES]), 0., 1e10)
ss.to_csv('submission1.csv', index=False)