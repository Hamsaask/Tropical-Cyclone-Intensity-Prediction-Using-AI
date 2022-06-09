# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:13:15 2021

@author: SKH
"""

import  xarray as xr 
import  pandas as pd
import numpy as np
import datetime

df=pd.read_csv('../Data/IMD_NEW.csv')
# df.drop(['YYYYMMDD','Sorting 0- 99', 'Level of TC', 'Rad','Tau','MSLP', 'windcode', 'rad1', 'rad2',
#            'rad3', 'rad4', 'radp', 'rrp', 'mrd', 'gust', 'eye', 'subregn',
#            'maxseas', 'intials', 'dir', 'speed ', 'name', 'depth', 'seas',
#            'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36',
#            'Unnamed: 37', 'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40'],axis='columns',inplace=True)
# df['Lat']=df["Lat"]/10
# df["Lon"]=df["Lon"]/10

df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df['Date'] = df['Date'].astype('datetime64[ns]')
# df.drop(['Date '],axis='columns',inplace=True)
# df['Date_f']=np.zeros(5404)
# df['Date_f'] = pd.to_datetime(df['YYYYmmddHH'], format="%Y%m%d%H")
# df['Date_f'] = df['Date_f'].astype('datetime64[ns]')
# df['year']=df['Date_f'].dt.year
# df['month']=df['Date_f'].dt.month
# df['dte']=df['Date_f'].dt.day
# df['hour']=df['Date_f'].dt.hour
# df['min']=df['Date_f'].dt.minute
# df['sec']=df['Date_f'].dt.second
# df['Date2']=np.zeros(5404)
# #df['Date2']=df['Date']
N=df.shape[0]
df['Date2']=np.zeros(N)
er=0
for i in range (0,N):
    try:
        # yr=df['year'][i]
        # mn=df['month'][i]
        # date=df['dte'][i]
        # hr=df['hour'][i]
        # mint=df['min'][i]
        # ss=df['sec'][i]
        # dt=datetime.datetime(yr,mn,date,hr,mint,ss)
        # dt64=np.datetime64(dt)
        # print(dt64)
        dt64=df['Date'][i]
        date_time_obj = datetime.datetime.strftime(dt64,"%Y-%m-%dT%H:%M:%S.%f")
        df['Date2'][i]=date_time_obj
        # dt=df['Date_f'][i]
        # df['Date_f'][i]=np.datetime64(dt)
        # # dt=df['Date_f'][i]
        # # df['Date_f'][i]=np.datetime64(dt)
        #print(df['Date2'][i])
    except:
        #print(df['Date_f'][i])
        print('error')
        er+=1
    



# df['Date2']=df['YYYYmmddHH']
# df['Date2']=df['Date2'].astype(int)
# df.drop(['year', 'month', 'dte', 'hour','min', 'sec'],axis='columns',inplace=True)


#df_ext = df[["Basin", "Cyc Num",'Date','Lat','Lon','Vmax']]
df_ext=df[['Sno','Date2','Lat','Lon','Vmax']]


df_ext['Lat_ext']=np.zeros(N)
df_ext['Lon_ext']=np.zeros(N)
df_ext['min MSLP']=np.zeros(N)

c=0
er=0


for i,row in df_ext.iterrows():
    try:
        #print('DONE')
        lt=row.Lat
        ln=row.Lon
        tt=row.Date2
        tt= datetime.datetime.strptime(tt,"%Y-%m-%dT%H:%M:%S.%f")
        tt=np.datetime64(tt)
   
        mslp= xr.open_dataset('../Data/mean_SLP.nc')
        #mslp=mslp.loc[dict(longitude=slice(ln-0.5,ln+0.5))]
 
        #mslp=mslp.loc[dict(latitude=slice(lt+0.5,lt-0.5),expver=1,time=tt)]
        
        mslp=mslp.loc[dict(expver=1,time=tt)]
        mslp_a=mslp['msl']
        #print(mslp_a)
        min_i=mslp_a.min(dim=['latitude','longitude']).values
        mslp_final=mslp.where(mslp['msl']==min_i,drop=True)
        lt_f=mslp_final.latitude.values
        ln_f=mslp_final.longitude.values
        if (lt_f.size>1):
            lt_f=lt_f.mean()
        if(ln_f.size>1):
            ln_f=ln_f.mean()
   
        df_ext["Lat_ext"][i]=lt_f
        df_ext["Lon_ext"][i]=ln_f
        df_ext['min MSLP'][i]=min_i
        c=c+1
        print('DONE')
    except:
        print('ERROR')
        er+=1
        continue
    


    
df_ext['Lat_error']=np.zeros(N)
df_ext['Lon_error']=np.zeros(N)

df_ext['Lat_error']=df_ext['Lat']-df_ext['Lat_ext']
df_ext['Lon_error']=df_ext['Lon']-df_ext['Lon_ext']
df_ext['Lat_error'].plot(xlim=[0,50],ylim=[-2,2])
df_ext['Lon_error'].plot(xlim=[0,50],ylim=[-2,2])

d=df_ext.loc[~(df_ext==0).any(axis=1)]
df_new=d[['Sno','Date2','Lat_ext','Lon_ext','Vmax']]





#df_new.to_csv('../Data/IMD_EXT.csv')
