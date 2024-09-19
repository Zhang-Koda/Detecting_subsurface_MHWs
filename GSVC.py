
import numpy as np
import math
import h5py  
import datetime
import scipy.io as scio  
import pandas as pd
import xarray as xr
import statsmodels.api as sm


def Weighted_regression(X1,X2,Y,weight):
    
    '''
    Enter the explanatory variables X1,X2 and the response variables Y, as well as the weight matrix weight. 
    Notice that their shape should be compatible.

    Return the coefficient of the GSVC model.

    '''
    assert X1.shape == X2.shape == Y.shape == weight.shape, "Input shapes are not compatible."

    # Data processing : straighten the data
    X1       =  X1.reshape(-1,1)
    X2       =  X2.reshape(-1,1)
    Y        =  Y.reshape(-1,1)
    weight   =  weight.reshape(-1,1)


    # Create a mask for NaN values
    nan_mask = ~np.isnan(X1) & ~np.isnan(X2) & ~np.isnan(Y) & ~np.isnan(weight)

    # Apply the mask to all variables
    X1       =  X1[nan_mask]
    X2       =  X2[nan_mask]
    Y        =  Y[nan_mask]
    weight   =  weight[nan_mask]
   
   
    # Define X & Y : assign weight
    X     =  np.column_stack((X1*np.sqrt(weight), X2*np.sqrt(weight), np.sqrt(weight)))
    Y     =  Y * np.sqrt(weight)
 
    
    # Create model
    mod = sm.OLS(Y,X)
    result = mod.fit()
    
    return result.params




def Calculate_beta_by_GSVC(lon_tar, lat_tar, bw_time, bw_space):

    '''
    Parameters
    ----------
    lon_tar: Target longitude (in degrees) at which the GSVC model is calculated.
    lat_tar: Target latitude (in degrees) at which the GSVC model is calculated.
    bw_time: Time bandwidth set by the GSVC model
    bw_space: Space bandwidth set by the GSVC model, in the meridional (north-south) direction


    Returns
    -------
    BETA : Coefficient of the GSVC model at the target latitude and longitude
    '''    

    
    # ------------------- 1 Experimental parameter setting  -------------------
    # the space resolution of data is 1°
    resolution = 1
    
    # Calculate the amount of data based on the bandwidth
    bw_time_num   =  2 * bw_time
    bw_space_num  =  int( 1/resolution * bw_space * 2 ) # in the meridional bandwidth
    
    
    
    # ---------- 2 Compute weight matrix : Gaussian kernel function  ----------
    index_time = 2 * bw_time_num + 1
    index_lon = 4 * bw_space_num + 1
    index_lat = 2 * bw_space_num + 1
       
    dataw = np.full((index_lon, index_lat, index_time), np.nan)
    wlon = np.linspace(-2, 2, index_lon)
    wlat = np.linspace(-2, 2, index_lat)
    wtime = np.linspace(-2, 2, index_time)
       
    # Gaussian weighting function
    exp_wlon = np.exp(-wlon**2)
    exp_wlat = np.exp(-wlat**2)
    
    for t in range(index_time):
        dataw[:, :, t] = np.exp(-wtime[t]**2) * np.outer(exp_wlon, exp_wlat)
    
    dataw[dataw < 0.01] = np.nan
    # Check if np.nan is of type float
    assert isinstance(np.nan, float)
    
    
    lon_ind = np.argmin(abs((lon_tar - XX)))
    lat_ind = np.argmin(abs((lat_tar - YY)))
    
    


    # ---------- 3. Estimate BETA  ----------------------------------------
    BETA  =   np.full([366,dim], np.nan)
    if np.any(np.isnan(SSTA[lon_ind,lat_ind,:])):
        print('land')
    else:
        xcoord =  np.arange(lon_ind - 2*bw_space_num,lon_ind + 2*bw_space_num+1,1)
        ycoord =  np.arange(lat_ind - 1*bw_space_num,lat_ind + 1*bw_space_num+1,1)
    
        start_date = str(TIME[0]).split('T')[0]
        end_date = str(TIME[-1]).split('T')[0]
        date_range = pd.date_range(start=start_date, end=end_date)
        dates_2012 = date_range.map(lambda x: x.replace(year=2012))
        day_of_year = np.array(dates_2012.map(lambda x: x.timetuple().tm_yday))
        time_idx = np.arange(len(day_of_year))
        
        # Used to store results
        BETA  =   np.full([366,dim], np.nan)
        
        for tt in range(366):
    
            center_day = np.where(day_of_year==tt+1)[0]
            target_day = []
            year_num=0
            for pos in center_day:
                start = pos-bw_time_num
                end = pos+bw_time_num+1
                if start>time_idx[0] and end<time_idx[-1]:
                    year_num=year_num+1
                    target_day.append(time_idx[start:end])
            target_day = np.array(target_day).reshape(-1)
        
        
            # Weight matrix for replicate samples
            dataw_rep = np.tile(dataw, (1, 1, year_num))
           
            Xc, Yc, Tc = np.meshgrid(xcoord, ycoord, target_day, indexing='ij')
            predictor1 = SSTA[Xc, Yc, Tc]
            predictor2 = SSHA[Xc, Yc, Tc]
            response   = STA[Xc, Yc, Tc]
    
    
            BETA[tt, :] = Weighted_regression(predictor1, predictor2, response, dataw_rep).flatten()
    return BETA


SSTA = xr.open_dataset('./SSTA.nc').SST.values.squeeze().transpose(2,1,0)
SSHA = xr.open_dataset('./SSHA.nc').SSH.values.squeeze().transpose(2,1,0)
STA = xr.open_dataset('./STA.nc').T_55.values.squeeze().transpose(2,1,0)

lon1 = xr.open_dataset('./SSTA.nc').lon.values.squeeze()
lat1 = xr.open_dataset('./SSTA.nc').lat.values.squeeze()
time1 = xr.open_dataset('./SSTA.nc').time.values.squeeze()
lon2 = xr.open_dataset('./SSHA.nc').lon.values.squeeze()
lat2 = xr.open_dataset('./SSHA.nc').lat.values.squeeze()
time2 = xr.open_dataset('./SSHA.nc').time.values.squeeze()
lon3 = xr.open_dataset('./STA.nc').lon.values.squeeze()
lat3 = xr.open_dataset('./STA.nc').lat.values.squeeze()
time3 = xr.open_dataset('./STA.nc').time.values.squeeze()

# Check that the latitude and longitude are same
lon_equal = (set(lon1) == set(lon2) == set(lon3))
lat_equal = (set(lat1) == set(lat2) == set(lat3))
time_eauql = (set(time1) == set(time2) == set(time3))
if lon_equal and lat_equal and time_eauql:
    LON  =  lon1
    LAT  =  lat1
    TIME = time1
else:
    print("The latitude and longitude of the data are inconsistent!")



dim=3
LON = np.arange(-165.375,-115.375,1)
LAT = np.arange(-28.625,-2.375,1)
BETA_re =np.full([len(LON), len(LAT), 366, dim], np.nan)
for i in range(len(LON)):
    for j in range(len(LAT)):
        BETA_re[i,j,:,:] = Calculate_beta_by_GSVC(LON[i],LAT[j], 30, 1)








