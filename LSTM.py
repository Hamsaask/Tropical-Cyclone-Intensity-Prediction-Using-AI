# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 19:13:56 2022

@author: Hamsaa SK
"""

import xarray as xr
import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy import stats
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras import initializers
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import concatenate
from tensorflow.keras.layers import Bidirectional
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold
from sklearn.model_selection import KFold
import seaborn as sns
from scipy import stats
import numpy as np
from matplotlib.ticker import PercentFormatter
from tensorflow import keras 
from keras.models import load_model

plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

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

df=pd.read_csv('../../IMD_FINALMOST.csv')


#df['SH']=(df['SH1']+df['SH2']+df['SH3']+df['SH4'])/4

'''
df['Del_V']=np.zeros(df.shape[0])
for i in range(0,df.shape[0]-1):
    df['Del_V'][i]=df['Vmax'][i+1]-df['Vmax'][i]
    
'''   
# df.drop(df.shape[0]-1,axis='rows',inplace=True)

df=df[[ 'Sno','Date', 'Lat', 'Lon',
         'MSLP',  'VOR', 'TP',
       'SH1','SH2','SH3','SH4', 'MSR','MW1','MW2','Vmax']]



full_columns=df.columns

#df.set_index('Cyc Num')
df['Date']=pd.to_datetime(df['Date'])
df['Date']=df['Date'].dt.strftime("%Y%m%d%H")
df['Date']=pd.to_numeric(df['Date'])
df=df[(df.Vmax != 'bfill')]
df['Vmax']=pd.to_numeric(df['Vmax'])
# Plotting each distribution

#mask = (df['Date2'] > '2019113018')
#df_validate=df[mask]
#mask2 = (df['Date2'] <'2019113018')
#df=df[mask2]
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


'''
for i in range(1, df.shape[1]):
    plt.plot(df.values[:, i:i+1])
    plt.title(full_columns[i] + ', ' + df.columns[i])
    plt.show()
'''



reframed = series_to_supervised(df, 1, 12)
reframed=reframed[['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)',
       'var6(t-1)', 'var7(t-1)', 'var8(t-1)', 'var9(t-1)', 'var10(t-1)',
       'var11(t-1)', 'var12(t-1)', 'var13(t-1)','var14(t-1)','var15(t-1)', 'var1(t)', 'var2(t)',
       'var3(t)', 'var4(t)', 'var5(t)', 'var6(t)', 'var7(t)', 'var8(t)',
       'var9(t)', 'var10(t)', 'var11(t)', 'var12(t)', 'var13(t)','var14(t)','var15(t)','var15(t+11)']]





df_val=reframed
tn = int(df_val.shape[0]*0.8)
train = df_val[:tn]
test = df_val[tn:]
X_train=train.drop(['var15(t+11)'],axis='columns')
y_train=train['var15(t+11)'].values.reshape(-1,1)
X_test=test.drop(['var15(t+11)'],axis='columns')
y_test=test['var15(t+11)'].values.reshape(-1,1)



scaler = MinMaxScaler(feature_range=(0,1))
X_train  = scaler.fit_transform(X_train)
X_test  = scaler.fit_transform(X_test)


timesteps=1
dim1=int(X_train.shape[1])
dim2=int(X_test.shape[1])

X_train = X_train.reshape((X_train.shape[0], timesteps, dim1))
X_test = X_test.reshape((X_test.shape[0], timesteps, dim2))

y_train = y_train.reshape((y_train.shape[0], 1, 1))
y_test = y_test.reshape((y_test.shape[0], 1, 1))


# LSTM model Hyperparametrs

#optim=['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
#epoch=[10,50,100,150,200]
#b_s=[10,20,40,60,80,100]
#nodes = [100,200,500]

nodes=[150]
dr_ly = [0.20]
#optim = ['Adam']
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
#RMSprop
optim=[opt]
activ = ['relu']
epoch = [200]
b_s = [20]
input=(X_train.shape)
train_in = X_train
train_out = y_train
test_in = X_test
test_out = y_test
print('OVER')

kf = KFold(n_splits=10)


data=[]
initializer = tensorflow.keras.initializers.Orthogonal()
#initializer = tensorflow.keras.initializers.GlorotNormal()
bias_initializer = tensorflow.keras.initializers.HeNormal()
#bias_initializer = 'Zeros'
#for train_index, test_index in kf.split(X_train):
    
for unit in nodes:
   for dropout in dr_ly:
       for activation in activ:
           for optimizer in optim:
               for epochs in epoch:
                   for batch in b_s:
                            #print("TRAIN:", train_index, "TEST:", test_index)
                            #X_train1, X_test1 = X_train[train_index[:-1]], X_train[test_index[:-1]]
                            #y_train1, y_test1 = y_train[train_index[:-1]], y_train[test_index[:-1]]
                            
                            
                            model = Sequential()
                            model.add(Bidirectional(LSTM(unit, input_shape=input)))
                            model.add(Dropout(dropout))
                            model.add(Dense(1, kernel_initializer=initializer,use_bias=False, bias_initializer=bias_initializer))
                            model.add(Activation(activation))
                            model.compile(loss='mae', optimizer=optimizer)
                        
                            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(test_in, test_out),verbose=2, shuffle=False)


                            pred_t = model.predict(X_train)
                            pred = model.predict(X_test)
                            pred=pred.squeeze()
                            y_test=y_test.squeeze()
                            y_train=y_train.squeeze()
                        
                            rmse=sqrt(mean_squared_error(y_test,pred))
                            rmse_t=sqrt(mean_squared_error(y_train,pred_t))
                            data.append([unit,dropout,activation,optimizer,epochs,batch,rmse,rmse_t])
                            print('RMSE for',unit,dropout,activation,optimizer,epochs,batch,'IS: /n',rmse_t,rmse)
                            
                            plt.plot(history.history['loss'])
                            plt.plot(history.history['val_loss'])
                            plt.title('Train and Test loss for 72hr forecast')
                            plt.ylabel('Loss (MAE)')
                            plt.xlabel('Epoch')
                            plt.legend(['Train data','Test data'], loc='upper right')
                            plt.text(11, 21, 'Train RMSE:'+str(rmse_t)+'m/s', fontsize = 10)
                            plt.text(11,20,"Test RMSE: "+str(rmse)+'m/s',fontsize=10)
                            plt.show()
                            
                            
                            

                            
cols=['Nodes','Dropout','Activation','Optimizer','Epochs','Batch','Test RMSE','Train_RMSE']
error_df=pd.DataFrame(data,columns=cols)
print(model.summary())





'''
import pickle
pickle.dump(model,open('lstm_tuned.pkl','wb'))
'''
#load_model=pickle.load(open('lstm.pkl','rb'))
#result = load_model.predict(X_val)
#print(sqrt(mean_squared_error(y_val,result)))

'''                 
optimizer=['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
epochs=[10,50,100,150]
batchsize=[10,20,40,60,80,100]
param_grid=dict(batch_size=batchsize,epochs=epochs,optimizer=optimizer)
grid=GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=7,cv=3)
grid_result=grid.fit(X_train,y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))             



plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Train loss ')
plt.ylabel('Loss (MAE)')
plt.xlabel('Epoch')
plt.legend(['train'], loc='center right')
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(' Model Accuracy: Training versus Validation')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='center right')
plt.show()

'''


#model.save('lstmimd66.h5')

'''
df['Change']=np.zeros(df.shape[0])
for i in range(0,df.shape[0]-4):
    df['Change'][i]=df['Vmax'][i+4]-df['Vmax'][i]
mask=(df['Change']>=25)
df=df[mask]
df.drop('Change',axis='columns',inplace =True)
print(df
'''

df_validate=pd.read_csv('../../JTWC_FINALMOST_VAL.csv')
df_validate['Date2']=pd.to_datetime(df_validate['Date2'])
df_validate['Date2']=df_validate['Date2'].dt.strftime("%Y%m%d%H")
df_validate['Date2']=pd.to_numeric(df_validate['Date2'])
# mask = (df['Sno'] > 179)
# df_validate=df[mask]

df_validate=df_validate[[ 'Cyc Num','Date2', 'Lat_ext', 'Lon_ext',
         'MSLP',  'VOR', 'TP',
       'SH1','SH2','SH3','SH4', 'MSR','MW1','MW2','Vmax']]



ref_val=series_to_supervised(df_validate, 1,2)
tt=ref_val['var2(t+1)']
c=ref_val['var1(t+1)']

ref_val=ref_val[['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)',
       'var6(t-1)', 'var7(t-1)', 'var8(t-1)', 'var9(t-1)', 'var10(t-1)',
       'var11(t-1)', 'var12(t-1)', 'var13(t-1)','var14(t-1)','var15(t-1)', 'var1(t)', 'var2(t)',
       'var3(t)', 'var4(t)', 'var5(t)', 'var6(t)', 'var7(t)', 'var8(t)',
       'var9(t)', 'var10(t)', 'var11(t)', 'var12(t)', 'var13(t)','var14(t)','var15(t)','var15(t+1)']]


X_val=ref_val.drop('var15(t+1)',axis='columns')
y_val=ref_val['var15(t+1)']
scaler = MinMaxScaler(feature_range=(0,1))
X_val  = scaler.fit_transform(X_val)
X_val=X_val.reshape((X_val.shape[0],1,X_val.shape[1]))
load_=load_model('LSTM_NEW/lstm12.h5')
val_pred=load_.predict(X_val)
val_pred=val_pred.squeeze()
rmse_tt=sqrt(mean_squared_error(y_val,val_pred))
y_val=y_val.values
err_m= np.mean(val_pred-y_val)
l=stats.pearsonr(val_pred, y_val)
corr=l[0]

print(corr,rmse_tt,err_m)

'''
sns.scatterplot(x=y_val,y=val_pred,hue=val_pred)
plt.legend(title='Intensity Range (m/s) ',loc='lower right')
plt.xlabel('Actual Vmax(m/s)')
plt.ylabel('Predicted Vmax(m/s)')
plt.title('12h Predicted vs Actual Vmax')
'''

'''
rmse_tt=sqrt(mean_squared_error(y_val,val_pred))

y_val=y_val.values
err_m= np.mean(val_pred-y_val)
l=stats.pearsonr(val_pred, y_val)
corr=l[0]
g=sns.scatterplot(y_val,val_pred)
g.set(xlabel='Actual data (m/s) ', ylabel='48hr Forecast data (m/s)')
#sns.lmplot(x=y_val, y=val_pred)
plt.title('Scatter plot between 48hr forecast and actual data  (2017-2019) ')
plt.text(15, 110, 'RMSE:'+str(rmse_tt)+'m/s', fontsize = 10)#(0,120)
plt.text(15, 100, 'Correlation:'+str(corr), fontsize = 10)#(0,120)
plt.text(15, 90, 'Bias:'+str(err_m)+'m/s', fontsize = 10)#(0,120)

'''



plt.plot(y_val)
plt.plot(val_pred)
plt.title('Predicted vs actual Vmax - 12hr forecast')
plt.ylabel('Vmax')
plt.xlabel('Index')
plt.legend(['Actual', 'Validate'], loc='upper right')
#plt.text(0, 140, 'RMSE:'+str(rmse)+'m/s', fontsize = 10)#(0,120)
plt.show()




'''
df_validate1=pd.read_csv('../../IMD_FINALMOST_VAL.csv')
df_validate1['Date']=pd.to_datetime(df_validate1['Date'])
df_validate1['Date']=df_validate1['Date'].dt.strftime("%Y%m%d%H")
df_validate1['Date']=pd.to_numeric(df_validate1['Date'])
# mask = (df['Sno'] > 179)
# df_validate=df[mask]
df_validate1=df_validate1[[ 'Sno','Date', 'Lat', 'Lon',
         'MSLP',  'VOR', 'TP',
       'SH1','SH2','SH3','SH4', 'MSR','MW1','MW2','Vmax']]

ref_val1=series_to_supervised(df_validate1, 1,13)
tt=ref_val1['var2(t+12)']
c=ref_val1['var1(t+12)']
ref_val1=ref_val1[['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)',
       'var6(t-1)', 'var7(t-1)', 'var8(t-1)', 'var9(t-1)', 'var10(t-1)',
       'var11(t-1)', 'var12(t-1)', 'var13(t-1)','var14(t-1)','var15(t-1)', 'var1(t)', 'var2(t)',
       'var3(t)', 'var4(t)', 'var5(t)', 'var6(t)', 'var7(t)', 'var8(t)',
       'var9(t)', 'var10(t)', 'var11(t)', 'var12(t)', 'var13(t)','var14(t)','var15(t)','var15(t+12)']]
X_val1=ref_val1.drop('var15(t+12)',axis='columns')
y_val1=ref_val1['var15(t+12)']
scaler = MinMaxScaler(feature_range=(0,1))
X_val1  = scaler.fit_transform(X_val1)
X_val1=X_val1.reshape((X_val1.shape[0],1,X_val1.shape[1]))
load_=load_model('LSTM_NEW/lstmimd72.h5')
val_pred1=load_.predict(X_val1)
val_pred1=val_pred1.squeeze()
rmse_tt=sqrt(mean_squared_error(y_val1,val_pred1))
y_val1=y_val1.values
err_m= np.mean(val_pred1-y_val1)
l=stats.pearsonr(val_pred1, y_val1)
corr=l[0]
print(corr,rmse_tt,err_m)

'''

'''
ax=sns.scatterplot(x=y_val1,y=val_pred1,hue=val_pred1)

plt.legend(title='Intensity Range (m/s) ',loc='lower right')
plt.xlabel('Actual Vmax(m/s)')
plt.ylabel('Predicted Vmax(m/s)')
plt.title('12h Predicted vs Actual Vmax')
'''
err=val_pred-y_val
#err1= val_pred1-y_val1

'''

g=sns.displot(data={'Jtwc':err,'Imd':err1},kde=True,bins=50)
#g.yaxis.set_major_formatter(PercentFormatter(100))
g.set(xlabel='Bias (m/s) ', ylabel='Density')
#plt.axvline(x = np.mean(err)-2, ymin=0, ymax=0.7  , linestyle='-')
plt.title('Bias Distribution plot  (2017 - 2019)  for 48hr forecast')
plt.show()

'''




#tt=tt-(tt[0:1].values)
c=c.astype(int)
sns.set_style("whitegrid", {'axes.grid' : False})
g=sns.boxplot(c,err)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.grid(False)
g.set(xlabel='Cyclone number ', ylabel='Error between predicted and actual Vmax(m/s)')
g.set_ylim(-80,80)
plt.axhline(y=0 , linestyle='-')
plt.title('Bias Box Plot  (2017 - 2019) for 72hr forecast ')
plt.show()



'''
g=sns.violinplot(tt,err)
g.set_xticklabels(g.get_xticklabels(),rotation=30)
g.set(xlabel='Cyclone number', ylabel='Error between predicted and actual Vmax(m/s)')
plt.axhline(y=0, linestyle='-')
plt.title(' Bias Violin plot (2017-2019) for 48hr forecast ')
plt.show()
'''

'''

import matplotlib.pyplot as plt
import matplotlib
#matplotlib.rcParams['legend.handlelength'] = 1

hrs=[6,12,18,24,30,36,42,48,54,60,66,72]


c=[ 0.98,0.95,0.91,0.84,0.72,0.66,0.63,0.61,0.56,0.52,0.45,0.38]
r=[ 8.46,9.16,10.14,12.83,16.92,18.50,19.65,20.55,21.56,22.72,24.33,25.97]
b=[ 3.32,1.9,1.22,-2.22,-4.24,-5.59,-7.75,-9.71,-10.49,-11.88,-13.47,-15.31]

c1=[ 0.97,0.94,0.91,0.85,0.7,0.62,0.56,0.51,0.48,0.45,0.39,0.36]
r1=[ 11.86,12.70,14.43,18.6,28.06,32.01,34.23,35.78,36.16,36.67,37.86,37.65]
b1=[ -9.84,-7.90,-8.4,-11.01,-18.65,-21.95,-23.77,-24.69,-24.80,-25.09,-25.86,-25.36]



plt.scatter(hrs,c,marker='o',edgecolors='face')
#plt.plot(hrs,c)
plt.scatter(hrs,c1,marker='o',edgecolors='face')
#plt.plot(hrs,c1)
plt.ylabel('Correlation between actual and forecast data')
plt.xlabel('Forecast time (hours) ')
plt.xticks(hrs,['6','12','18','24','30','36','42','48','54','60','66','72'])
plt.legend(['IMD','JTWC'])
plt.plot(hrs,c)
plt.plot(hrs,c1)
plt.title('Correlation between actual and forecast data')

plt.scatter(hrs,r,marker='o')
plt.scatter(hrs,r1,marker='o')
plt.ylabel('RMSE between actual and forecast data')
plt.xlabel('Forecast time (hours) ')
plt.xticks(hrs,['6','12','18','24','30','36','42','48','54','60','66','72'])
plt.legend(['IMD','JTWC'])
plt.plot(hrs,r)
plt.plot(hrs,r1)
plt.title('RMSE (m/s)  between actual and forecast data')
plt.show()

plt.scatter(hrs,b)

plt.scatter(hrs,b1)

plt.ylabel('Bias between actual and forecast data')
plt.xlabel('Forecast time (hours) ')
plt.xticks(hrs,['6','12','18','24','30','36','42','48','54','60','66','72'])
plt.legend(['IMD','JTWC'])
plt.title('Bias (m/s)  between actual and forecast data')
plt.plot(hrs,b)
plt.plot(hrs,b1)
plt.gca().invert_yaxis()
plt.show()



#Bias between model comparison 


'''
