import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.date_time = pd.to_datetime(train.date_time)
train['month'] = train['date_time'].dt.month
train['week'] = train['date_time'].dt.isocalendar().week
train['is_weekend'] = (train["date_time"].dt.dayofweek >= 5).astype("int")
train['hour'] = train['date_time'].dt.hour
df_data = train.set_index('date_time')
df_data.to_csv('train1.csv', index=False)

test.date_time = pd.to_datetime(test.date_time)
test['month'] = test['date_time'].dt.month
test['week'] = test['date_time'].dt.isocalendar().week
test['is_weekend'] = (test["date_time"].dt.dayofweek >= 5).astype("int")
test['hour'] = test['date_time'].dt.hour
df_data = test.set_index('date_time')
df_data.to_csv('test1.csv', index=False)
