#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19 Dec, 2024

@author: jason

replicates the start of Box et al (2009)
    instead of RACMO as training target, uses 2.5 km CARRA data

"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from numpy.polynomial.polynomial import polyfit
from scipy import stats

base_path='/Users/jason/Dropbox/S3/recon_CARRA/'
os.chdir(base_path)

carra_path='/Users/jason/Dropbox/temp/T_recon/CARRA/t2m/'
training_output_path='/Users/jason/Dropbox/temp/T_recon/training_output/'
fn='./CARRA_ancil/CARRA_west_mask_Greenland.tif'
dx = rasterio.open(fn)
# profile=dx.profile
mask=dx.read(1)    

plt.close()
plt.imshow(mask)
plt.axis('off')
plt.colorbar()
plt.show()
#%%

# print(np.shape(mask))

iyear=1991 ; fyear=2020
n_years=fyear-iyear+1

years=np.arange(iyear,fyear+1).astype(str)


ni,nj=1427, 1246 # CARRA grid dimensionss

station_id=['4221','4250','4272','4360','4211']
station_name=['Ilulissat','Nuuk','Qaqortoq','Tasiilaq','Upenavik']
months=['Feb','Jul']
# months=['Jul']
monthnums=['02','07']

# simplify, select only the first element of the arrays above, test case
station_id=[station_id[3]]
station_name=[station_name[3]]
# months=[months[3]]

n_stations=len(station_id)
n_months=len(months)

# 3D array to compute stats on
statvar_y=np.zeros((n_years,ni,nj))

compvar='t2m'

for station_index,station in enumerate(station_id):
    df=pd.read_csv(f'./station_data/ASCII/{station}.csv')
    v=np.where((df.Year>=iyear)&(df.Year<=fyear))
    for mm,month in enumerate(months):
        statvar_x=np.zeros(n_years)
        statvar_x[:]=df[month][v[0]]
        for yy,year in enumerate(years):
        
            print('loading years into 3D array',year)
            
            fn=f'{carra_path}/{month}/t2m_{year}_{monthnums[mm]}.tif'
            dx = rasterio.open(fn)
            profile=dx.profile
            d=dx.read(1)    
            statvar_y[yy,:,:]=d


#%% this step is relatively slow, ~60 sec if mask>0, slower if not masking the domain

        ni,nj=1427, 1246
        
        slopes=np.zeros((ni,nj))
        intercepts=np.zeros((ni,nj))
        corrs=np.zeros((ni,nj))
        confidence=np.zeros((ni,nj))
        
        x=statvar_x.copy()
        v2=np.where(np.isfinite(x))
        
        for i in range(ni-1):
            print(f'{month} {station} countdown: {ni-i}')
            for j in range(nj-1):
                if mask[i,j]>0: # only Greenland
                    y=statvar_y[:,i,j]
                    x2=x[v2]
                    y2=y[v2]
                    # print(x2)
                    # print(y2)
                    coefs=stats.pearsonr(x2,y2)
                    b, m = polyfit(x2,y2, 1)
                    # print(coefs)
                    # print(m)
                    slopes[i,j]=m
                    intercepts[i,j]=b
                    corrs[i,j]=coefs[0]
                    confidence[i,j]=1-coefs[1]
        
        #%%
        plotvar=corrs.copy()
        plotvar=slopes.copy()
        plotvar[((confidence<0.8)&(mask>0))]=-1
        plt.close()
        plt.imshow(plotvar,cmap='seismic',vmin=-1,vmax=1)
        plt.axis('off')
        plt.title(f'{station_name[station_index]} station {month}')
        plt.colorbar()
        plt.show()
        
        #%% write out grids
        
        # get projection profile
        fn='./CARRA_ancil/ancil/CARRA_west_mask_Greenland.tif'
        dx = rasterio.open(fn)
        profile=dx.profile
        
        v=slopes==0
        corrs[v]=np.nan
        confidence[v]=np.nan
        intercepts[v]=np.nan
        slopes[v]=np.nan
        
        
        def write_tif(fn,var):
            with rasterio.Env():
                with rasterio.open(fn, 'w', **profile) as dst:
                    dst.write(var, 1)
            return None
        
        write_tif(f'{training_output_path}/cor_{station}_{month}.tif',corrs)
        write_tif(f'{training_output_path}/b_{station}_{month}.tif',intercepts)
        write_tif(f'{training_output_path}/m_{station}_{month}.tif',slopes)
        write_tif(f'{training_output_path}/conf_{station}_{month}.tif',confidence)
        
        