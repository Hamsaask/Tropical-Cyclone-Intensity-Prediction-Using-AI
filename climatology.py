# -*- coding: utf-8 -*-
"""
Created on 10/11/2021

@author: hamsaa
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

  
data=xr.open_mfdataset('../Trial_wrf_cape/*.nc',combine='nested',concat_dim='center',parallel=True)
##data = data.where(data['v'] != -9999.)  
dm=data['cape'].mean(dim='center',skipna=True).squeeze().values



#dm=data['cape'][0,:,:].squeeze().values

#r=data['r_val'].mean(dim='center',skipna=True).squeeze().values
#t=data['theta_val'].mean(dim='center',skipna=True).squeeze().values
r=data['r_val'].squeeze().values
t=data['theta_val'].squeeze().values
# dm=dm['sst'].values
# t=t.values
t=np.radians(t)
# r=r.values

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))


ax.set_xticklabels([0,315,270,225,180,135,90,45])
ax.set_yticklabels(['','','','','300km'])
ax.set_theta_zero_location("N")
ax.set_rlim(bottom=0, top=300)

cont=ax.contourf(t,r,np.flipud(r))
plt.colorbar(cont)
plt.yticks(ha='right')
plt.title('mean - 1982 to 2019 in polar coordinates and rotated ')
# plt.savefig('SST')

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

ax.set_xticklabels([0,315,270,225,180,135,90,45])
ax.set_yticklabels(['','','','','','300km'])
ax.set_theta_zero_location("N")
ax.set_rlim(bottom=0, top=300)

cont1=ax.contourf(t,r,np.flipud(dm),cmap='plasma_r')
plt.colorbar(cont1)
plt.yticks(ha='right')
plt.title('CAPE wrf mean')

