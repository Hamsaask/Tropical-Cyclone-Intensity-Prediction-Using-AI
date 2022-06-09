# -*- coding: utf-8 -*-
"""
Created on 10/11/2021

@author: hamsaa
"""

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow import keras 
from keras.models import load_model
import numpy as np
from scipy import stats
import seaborn as sns 

# transform a time series dataset into a supervised learning dataset

def series_to_supervised(df, n_in, n_out, dropnan=True):
        n_vars = 1 if type(df) is list else df.shape[1]
        df = pd.DataFrame(df)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                else:
                        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
                agg.dropna(inplace=True)
        return agg
    


df = read_csv('../../JTWC_FINALMOST.csv')
df=df[[ 'Cyc Num','Date2', 'Lat_ext', 'Lon_ext',
         'MSLP',  'VOR', 'TP',
       'SH1','SH2','SH3','SH4', 'MSR','MW1','MW2','Vmax']]
full_columns=df.columns
df['Date2']=pd.to_datetime(df['Date2'])
df['Date2']=df['Date2'].dt.strftime("%Y%m%d%H")
df['Date2']=pd.to_numeric(df['Date2'])
df=df[(df.Vmax != 'bfill')]
df['Vmax']=pd.to_numeric(df['Vmax'])
df=df[~(df == 0). all(axis=1)]
df['MSLP'] = df['MSLP'].fillna((df['MSLP'].mean()))
df['VOR'] = df['VOR'].fillna((df['VOR'].mean()))
df['TP'] = df['TP'].fillna((df['TP'].mean()))
df['SH1'] = df['SH1'].fillna((df['SH1'].mean()))
df['SH2'] = df['SH2'].fillna((df['SH2'].mean()))
df['SH3'] = df['SH3'].fillna((df['SH3'].mean()))
df['SH4'] = df['SH4'].fillna((df['SH4'].mean()))
df['MSR'] = df['MSR'].fillna((df['MSR'].mean()))
df['MW1'] = df['MW1'].fillna((df['MW1'].mean()))
df['MW2'] = df['MW2'].fillna((df['MW2'].mean()))
df['Vmax'] = df['Vmax'].fillna((df['Vmax'].mean()))

# transform the time series data into supervised learning
data = series_to_supervised(df, 1,3)
data=data[['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)',
       'var6(t-1)', 'var7(t-1)', 'var8(t-1)', 'var9(t-1)', 'var10(t-1)',
       'var11(t-1)', 'var12(t-1)', 'var13(t-1)','var14(t-1)','var15(t-1)', 'var1(t)', 'var2(t)',
       'var3(t)', 'var4(t)', 'var5(t)', 'var6(t)', 'var7(t)', 'var8(t)',
       'var9(t)', 'var10(t)', 'var11(t)', 'var12(t)', 'var13(t)','var14(t)','var15(t)','var15(t+2)']]

df_val=data
tn = int(df_val.shape[0]*0.8)
train = df_val[:tn]
test = df_val[tn:]
X_train=train.drop(['var15(t+2)'],axis='columns')
y_train=train['var15(t+2)'].values.reshape(-1,1)
X_test=test.drop(['var15(t+2)'],axis='columns')
y_test=test['var15(t+2)'].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
X_train  = scaler.fit_transform(X_train)
X_test  = scaler.fit_transform(X_test)
timesteps=1
dim1=int(X_train.shape[1])
dim2=int(X_test.shape[1])

'''
X_train = X_train.reshape((X_train.shape[0], timesteps, dim1))
X_test = X_test.reshape((X_test.shape[0], timesteps, dim2))
y_train = y_train.reshape((y_train.shape[0], 1, 1))
y_test = y_test.reshape((y_test.shape[0], 1, 1))
'''

from sklearn.svm import SVR
model = SVR(kernel = 'rbf',epsilon=0.1)
#model = RandomForestRegressor(n_estimators=1000)
history=model.fit(X_train, y_train)
yhat = model.predict(X_test)
yhat_t=model.predict(X_train)
rmse=sqrt(mean_squared_error(y_test,yhat))
rmse_t=sqrt(mean_squared_error(y_train,yhat_t))
print('RMSE for','IS: /n',rmse_t,rmse)

#validation process


df_validate1=pd.read_csv('../../JTWC_FINALMOST_VAL.csv')
df_validate1['Date2']=pd.to_datetime(df_validate1['Date2'])
df_validate1['Date2']=df_validate1['Date2'].dt.strftime("%Y%m%d%H")
df_validate1['Date2']=pd.to_numeric(df_validate1['Date2'])

df_validate1=df_validate1[[ 'Cyc Num','Date2', 'Lat_ext', 'Lon_ext',
         'MSLP',  'VOR', 'TP',
       'SH1','SH2','SH3','SH4', 'MSR','MW1','MW2','Vmax']]

ref_val1=series_to_supervised(df_validate1, 1,3)
tt=ref_val1['var2(t+2)']
c=ref_val1['var1(t+2)']
ref_val1=ref_val1[['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)',
       'var6(t-1)', 'var7(t-1)', 'var8(t-1)', 'var9(t-1)', 'var10(t-1)',
       'var11(t-1)', 'var12(t-1)', 'var13(t-1)','var14(t-1)','var15(t-1)', 'var1(t)', 'var2(t)',
       'var3(t)', 'var4(t)', 'var5(t)', 'var6(t)', 'var7(t)', 'var8(t)',
       'var9(t)', 'var10(t)', 'var11(t)', 'var12(t)', 'var13(t)','var14(t)','var15(t)','var15(t+2)']]
X_val1=ref_val1.drop('var15(t+2)',axis='columns')

y_val1=ref_val1['var15(t+2)']
scaler = MinMaxScaler(feature_range=(0,1))
X_val1  = scaler.fit_transform(X_val1)

val_pred1=model.predict(X_val1)
val_pred1=val_pred1.squeeze()

rmse_v=sqrt(mean_squared_error(val_pred1,y_val1))
err_m= np.mean(val_pred1-y_val1)
l=stats.pearsonr(val_pred1, y_val1)
corr=l[0]



# plt.plot(y_test)
# plt.plot(yhat)
# plt.title('Predicted vs actual Vmax - 24hr forecast')
# plt.ylabel('Vmax')
# plt.xlabel('Time Step')
# plt.legend(['Actual', 'Test'], loc='upper right')
# plt.text(110, 110, 'Test RMSE:'+str(rmse)+'m/s', fontsize = 10) #(0,120)
# #plt.text(0, 140, 'RMSE:'+str(rmse)+'m/s', fontsize = 10)#(0,120)
# plt.show()

print(rmse,err_m,corr)



ax=sns.scatterplot(x=y_val1,y=val_pred1,hue=val_pred1)
sns.lineplot(x=y_val1,y=val_pred1,color='purple')
plt.legend(title='Intensity Range (m/s) ',loc='lower right')
plt.xlabel('Actual Vmax(m/s)')
plt.ylabel('Predicted Vmax(m/s)')
plt.title('12h Predicted vs Actual Vmax')

                            










