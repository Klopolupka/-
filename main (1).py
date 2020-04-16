import pandas as pd
import xgboost as xgb
import time
import os

transactions_train=pd.read_csv('data/transactions_train.csv')
train_target=pd.read_csv('data/train_target.csv')

#agg_features=transactions_train.groupby('client_id')['amount_rur', 'trans_date'].agg(['sum','mean','std','min','max']).reset_index()
agg_features=transactions_train.groupby('client_id')['amount_rur'].agg(['sum','mean','std','min','max']).reset_index()
agg_features_date=transactions_train.groupby('client_id')['trans_date'].agg(['mean']).reset_index()

counter_df_train=transactions_train.groupby(['client_id','small_group'])['amount_rur'].count()
cat_counts_train=counter_df_train.reset_index().pivot(index='client_id', \
                                                      columns='small_group',values='amount_rur')
cat_counts_train=cat_counts_train.fillna(0)
cat_counts_train.columns=['small_group_'+str(i) for i in cat_counts_train.columns]

train=pd.merge(train_target,agg_features,on='client_id')
train=pd.merge(train,agg_features_date,on='client_id')
train=pd.merge(train,cat_counts_train.reset_index(),on='client_id')

transactions_test=pd.read_csv('data/transactions_test.csv')

test_id=pd.read_csv('data/test.csv')

#agg_features_test=transactions_test.groupby('client_id')['amount_rur'].agg(['sum','mean','std','min','max']).reset_index()
agg_features_test=transactions_test.groupby('client_id')['amount_rur'].agg(['sum','mean','std','min','max']).reset_index()
agg_features_date_test=transactions_test.groupby('client_id')['trans_date'].agg(['mean']).reset_index()
counter_df_test=transactions_test.groupby(['client_id','small_group'])['amount_rur'].count()
cat_counts_test=counter_df_test.reset_index().pivot(index='client_id', columns='small_group',values='amount_rur')
cat_counts_test=cat_counts_test.fillna(0)
cat_counts_test.columns=['small_group_'+str(i) for i in cat_counts_test.columns]

test=pd.merge(test_id,agg_features_test,on='client_id')
test=pd.merge(test,agg_features_date_test,on='client_id')
test=pd.merge(test,cat_counts_test.reset_index(),on='client_id')

common_features=list(set(train.columns).intersection(set(test.columns)))
y_train=train['bins']
X_train=train[common_features]
X_test=test[common_features]

param={'objective':'multi:softprob','num_class':4,'n_jobs':4,'seed':42}
model=xgb.XGBClassifier(**param, n_estimators=300)
model.fit(X_train,y_train)

pred=model.predict(X_test)

submission = pd.DataFrame({'bins': pred}, index=test.client_id)
submission.head()

current_timestamp = int(time.time())
submission_path = 'submissions/{}.csv'.format(current_timestamp)

if not os.path.exists('submissions'):
    os.makedirs('submissions')

print(submission_path)
submission.to_csv(submission_path, index=True)

#predictions = [round(value) for value in pred]
#print(predictions)

#print(pred)
