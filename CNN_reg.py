import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from d2l import torch as d2l
import xarray as xr
import scipy.io as scio
import pandas as pd



# choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()  # Clear cuda video memory





"""
# =============================================================================
# #============================function preparation=============================
# =============================================================================
"""

def MSE_cal(Y_hat, y):
    '''
    Calculate the mean squared error between predicted and true values.

    Parameters:
    - Y_hat : Predicted values.
    - y : True values.

    Returns:
    - Mean squared error.
    '''
    temp_data = torch.square(Y_hat - y)
    total = temp_data.numel()
    temp_data = torch.nan_to_num(temp_data, nan=0.0)
    mse = temp_data.sum() / total
    
    return mse


def standardization(input_data,comp_data):
    '''
    Standardize input_data using the mean and standard deviation from flattened.

    Parameters:
    - input_data : The data to be standardized.
    - comp_data : The data used to compute mean and standard deviation.

    Returns:
    - np.ndarray: The standardized data.
    '''
    if input_data.size == 0 or comp_data.size == 0:
        raise ValueError("Input arrays must not be empty.")
        
    mean_value = np.nanmean(comp_data.flatten())
    std_deviation = np.nanstd(comp_data.flatten())

    if std_deviation == 0:
        raise ValueError("Standard deviation is zero. Cannot perform standardization.")

    standardized_data = (input_data - mean_value) / std_deviation
    return standardized_data


def evaluation(net, data_iter, device=None): 
    '''
    Evaluate a neural network using mean squared error on a given dataset.

    Parameters:
    - net : The neural network model.
    - data_iter : The data iterator.
    - device : The device on which to perform computations.

    Returns:
    - The average mean squared error.
    '''
    if isinstance(net, nn.Module):
        net.eval() 
    
    if not device:
        device = next(iter(net.parameters())).device
        
        
    total_mse = 0.0
    num_batches = len(data_iter)
    with torch.no_grad():
        for i, (X, y) in enumerate(data_iter):
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            y_hat = y_hat.squeeze()
            total_mse += MSE_cal(y_hat, y)
            
    average_mse = total_mse / num_batches

    return average_mse
           

def data_generation(net, data_iter, lat, lon, filename):
    '''
    Generate data using a neural network and save it to a (.mat) file.

    Parameters:
    - net : The neural network model.
    - data_iter : The data iterator.
    - lat : Number of latitude points.
    - lon : Number of longitude points.
    - output_file : Path to save the output file.
    - device : The device on which to perform computations.

    Returns:
    - None
    '''
    if isinstance(net, nn.Module):
        net.eval() 
    
    data = np.empty((0,lat,lon))
    
    with torch.no_grad():
        for i, (X, y) in enumerate(data_iter):  
            X = X.to(device)
            y_hat = net(X)
            y_hat = y_hat.squeeze()
            y_hat = y_hat.cpu().numpy()
            data = np.append(data, y_hat,axis=0)

    scio.savemat(filename ,{'data':data}) 



def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)  #variance to remain the same
        
def train_CNN_model(net, train_iter, val_iter, num_epochs,save_path=None):    
    '''
    Train a Convolutional Neural Network (CNN) model.
    
    Parameters:
    - net : The CNN model.
    - train_iter : The training data iterator.
    - val_iter : The validation data iterator.
    - num_epochs : Number of training epochs.
    - save_path : The path to save model.

    
    Returns:
    - net : The trained CNN model.
    - Trn_mse : The mse on training data iterator.
    - Val_mse : The mse on the validation data iterator.
    
    '''
    
    net.apply(init_weights)
    print('training on', device)
    net.to(device)

    optimizer = torch.optim.Adadelta(net.parameters())
    
    loss = nn.MSELoss()
    timer = d2l.Timer()

    Trn_mse = np.empty(0)
    Val_mse = np.empty(0)

    for epoch in range(num_epochs):
        print('epoch:'+str(epoch))

        net.train()
        trn_mse = 0
        for i, (X, y) in enumerate(train_iter):  
        
            timer.start()
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            y_hat = y_hat.squeeze()
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()  
            
            with torch.no_grad():
                mse = MSE_cal(y_hat, y)
    
            timer.stop()
            trn_mse += mse
            
        trn_mse = trn_mse / len(train_iter)
        trn_mse = trn_mse.cpu().numpy()
        Trn_mse = np.append(Trn_mse, trn_mse)
        
        val_mse = evaluation(net, val_iter)
        val_mse = val_mse.cpu().numpy()
        Val_mse = np.append(Val_mse, val_mse)

        print(f'train mse {trn_mse:.3f}, '
              f'test1 mse {val_mse:.3f},')

    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f'Model weights saved to {save_path}')

    return  Trn_mse, Val_mse















"""
# =============================================================================
# #=================01 Read Data(training & validation datasets)===============
# =============================================================================
"""

depth = XX
Batch_size = 365


SSTA = xr.open_dataset('./SSTA.nc').SSTA.values.squeeze()
SSHA = xr.open_dataset('./SSHA.nc').SSHA.values.squeeze()
STA = xr.open_dataset('./STA.nc').STA.values.squeeze()

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




start_date = str(TIME[0]).split('T')[0]
end_date = str(TIME[-1]).split('T')[0]
date_range = pd.date_range(start=start_date, end=end_date)


years = np.arange(1993, 2021)
np.random.seed(42)
trn_years = np.random.choice(years, size=int(0.8 * len(years)), replace=False)
val_years = np.setdiff1d(years, trn_years)

trn_index = (date_range[date_range.year.isin(trn_years)]- pd.Timestamp('1993-01-01')).days
val_index = (date_range[date_range.year.isin(val_years)]- pd.Timestamp('1993-01-01')).days


#-------------------------------For training datasets-------------------------------
SSHA1 = SSHA[trn_index,:,:]
SSTA1  = SSTA[trn_index,:,:]
SSHA1 = standardization(SSHA1,SSHA1)
SSTA1 = standardization(SSTA1,SSTA1)

XX_TRN = np.stack((SSTA1,SSHA1),axis=1) 
XX_TRN = np.nan_to_num(XX_TRN)
YY_TRN = np.nan_to_num(STA[trn_index,:,:])

XX_TRN = torch.tensor(XX_TRN, dtype=torch.float32)
YY_TRN = torch.tensor(YY_TRN, dtype=torch.float32)


train_ids = TensorDataset(XX_TRN, YY_TRN) 
trn_iter = DataLoader(dataset=train_ids, batch_size=Batch_size, shuffle=True)



#-------------------------------For validation datasets-------------------------------
SSHA2 = SSHA[val_index,:,:]
SSTA2  = SSTA[val_index,:,:]
SSHA2 = standardization(SSHA2,SSHA2)
SSTA2 = standardization(SSTA2,SSTA2)

XX_VAL = np.stack((SSTA2,SSHA2),axis=1)    
YY_VAL = STA[val_index,:,:]

XX_VAL = np.nan_to_num(XX_VAL)
YY_VAL = np.nan_to_num(YY_VAL)

XX_VAL = torch.tensor(XX_VAL, dtype=torch.float32)
YY_VAL = torch.tensor(YY_VAL, dtype=torch.float32)

val_ids = TensorDataset(XX_VAL, YY_VAL) 
val_iter = DataLoader(dataset=val_ids, batch_size=Batch_size, shuffle=False)



'''
# =============================================================================
# #==============================02 Training CNN model=========================
# =============================================================================
'''

channel = 2
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, channel, len(LAT), len(LON))
    
net = torch.nn.Sequential(
    Reshape(), 
    nn.Conv2d(2, 16, kernel_size=3,padding=1), nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3,padding=1), nn.ReLU(),
    nn.Conv2d(32, 1, kernel_size=3,padding=1))

X = torch.rand(size=(1, channel, len(LAT), len(LON)), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

  
num_epochs = 100
Trn_mse, Val_mse = train_CNN_model(net, trn_iter, val_iter, num_epochs, 'sequential_model_'+str(depth)+'.pth')



filename = './Output/Y_'+str(depth)+'.mat'
data_generation(net, val_iter, len(LAT), len(LON), filename)  

scio.savemat('./Output/Trn_mse'+str(depth)+'.mat',{'Trn_mse':Trn_mse})  
scio.savemat('./Output/Val_mse'+str(depth)+'.mat',{'Val_mse':Val_mse})  

