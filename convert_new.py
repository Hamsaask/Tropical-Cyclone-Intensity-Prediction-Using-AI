# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import xarray as xr 
import pandas as pd
import numpy as np
from numpy import pi
# import matplotlib.pyplot as plt 
from math import radians,sqrt,degrees,isnan
from scipy import ndimage 
# from enterprise.DeckThree import cyclone
import pickle

def distance(lon1,lon2,lat1,lat2):  
    # Function to calculate distance 
    t1=(lon2-lon1)**2
    t2=(lat2-lat1)**2
    D=sqrt(t1+t2)
    D=D*111
    return(D)

def convert_polar_rotate(par):    
    [i,df] = par
    print(i)
    date=df['Date2'][i]
    
    
    date=pd.to_datetime(date)
    date=int(date.strftime("%Y%m%d%H"))
    file=str(date)+'.nc'
    
    print(file)
    
    data_val = xr.open_dataset('sst/'+file)
    data_val=data_val.sel(expver=1)
    df['Date2']= pd.to_datetime(df['Date2'])
    
    N=df.shape[0]
    
    lt1=df['Lat_ext'][i]  # Center points 
    ln1=df['Lon_ext'][i]
    tt1=df['Date2'][i]
        
   
    if(tt1==data_val['time'].values):
            if(df['Cyc Num'][i]==df['Cyc Num'][i+1]):    
                var_slice=data_val.loc[dict(longitude=slice(ln1-7,ln1+7),latitude=slice(lt1+7,lt1-7),time=tt1)]
                lt=var_slice['latitude']
                ln=var_slice['longitude']
                tt=var_slice['time']
                ls=lt.size
                ns=ln.size
                var_slice=var_slice.to_array()
                #print(var_slice[:,0,0])
                #creating new dataset and coordinates 
                data = np.zeros([ls,ns])
                lt_mat=np.zeros([ls,ns])
                ln_mat=np.zeros([ls,ns])
                dist=np.zeros([ls,ns])
                angle=np.zeros([ls,ns])
        
    
                for j in range (0,ls):
                    for k  in range (0,ns):
                        lt2=lt[j]
                        ln2=ln[k]
                        rv=distance(ln1,ln2,lt1,lt2) # calculating distance between center and latitude longitude points 
                        x=(ln2-ln1)
                        y=(lt2-lt1)
                        thetav=degrees(np.arctan2(y,x)) # calculating theta value wrt center
                        if(thetav<0):
                            thetav=thetav+360
                    
                        val=var_slice[:,j,k].values
                        #print(val)
                        ltv=lt[j]
                        lnv=ln[k]
                
                 
                 
                        #2D array with the r , theta and variable values
                        data[j,k]=val
                        lt_mat[j,k]=ltv
                        ln_mat[j,k]=lnv
                        dist[j,k]=rv
                        angle[j,k]=thetav
                
              
                
                #calculating max and min values of the dataset         
                dmin=np.nanmin(data)
                dmax=np.nanmax(data)
        
        
                #Replacing Nan values with -9999 
                data1=data.copy()
                data1[np.isnan(data1)]=-9999
        
        
        
                #Calculating direction of the cyclone with the next time step.         
                lt1_0=df['Lat_ext'][i+1]
                ln1_0=df["Lon_ext"][i+1]
                d1=(lt1_0-lt1)
                d2=(ln1_0-ln1)
        
                direc=degrees(np.arctan2(d1,d2))
                if ( direc<0):
                    direc=direc+360
        
        
                l=[ln1,lt1] # Center coordinates 
    
        
        
                # Rotating the data array with the direction of the cyclone to allign . 
                data1_rot=ndimage.rotate(data1,(90-direc),reshape=False)
        
                #Replacing the values beyong the ranges with nan values
                data1_rot[data1_rot<dmin]=np.nan
                data1_rot[data1_rot>dmax]=np.nan
    
                     
                #dataset with r and theta as dimensions                           
    
                ds = xr.Dataset(
                    data_vars=dict(sst=(["r", "theta"], data1_rot),
                                   ),
            
                    coords=dict(
                            r_val=(["r", "theta"], dist),
                            theta_val=(["r", "theta"], angle),
                            time=tt1,
                            center=l
                            ),
                    attrs=dict(
                            description="SST in cylindrical coordinates.",
                            ),
                    )   
        
                #Cropping further for 500km radius 
                mask_r = (ds.r_val<600)
                
                
                cropped_ds = ds.where(mask_r, drop=True)
                
            
                
                outfile=open('sst_croped/'+file[:-3]+'_cropped.pkl','wb')
                pickle.dump(cropped_ds,outfile)
                outfile.close()
                
    return(None)      
                
                
# import datetime as dt
# from glob import glob
from concurrent.futures import ProcessPoolExecutor
df=pd.read_csv('JTWC_ERA_FINAL.csv')
df['Date2']=pd.to_datetime(df['Date2'],yearfirst=True)
files = list(df.index)
inp=[[i,df] for i in files]
# if __name__ == '__main__':
with ProcessPoolExecutor(max_workers=96) as exe:
     results=exe.map(convert_polar_rotate,inp)

           
        
                

    
    
    
    
    
