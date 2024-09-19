import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


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


def obtain_confusion_matrix(net, data_iter, num_lat, num_lon):

    '''
    Obtain confusion matrix: A for TP, B for FN, C for FP, D for TN.

    Parameters:
    - net (nn.Module): The neural network model.
    - data_iter (DataLoader): The data iterator.
    - num_lat (int): The number of latitude values.
    - num_lon (int): The number of longitude values.
    - device (torch.device): The device on which to perform computations.

    Returns:
    - tuple: A tuple containing the elements of the confusion matrix (A, B, C, D).
    '''

    if isinstance(net, nn.Module):
        net.eval() 
    A = torch.zeros(num_lat, num_lon, device=device)
    B = torch.zeros(num_lat, num_lon, device=device)
    C = torch.zeros(num_lat, num_lon, device=device)
    D = torch.zeros(num_lat, num_lon, device=device)

    with torch.no_grad():
        for i, (X, y) in enumerate(data_iter):
            X = X.to(device)
            Y_true = y.to(device)
            y_hat = net(X)

            probabilities = torch.sigmoid(y_hat)
            Y_pred = (probabilities[:, 0, :, :] <= probabilities[:, 1, :, :]).int()

            temp = Y_true - Y_pred
            A += torch.sum((temp == 0) & (Y_true == 1), dim=0)
            B += torch.sum((temp == -1), dim=0)
            C += torch.sum((temp == 1), dim=0)
            D += torch.sum((temp == 0) & (Y_true == 0), dim=0)
    return A, B, C, D   

def evaluate_accuracy_gpu(net, data_iter, device=None): 
    '''
    Evaluate the accuracy of a neural network model on a dataset using GPU acceleration.

    Parameters:
    - net (nn.Module): The neural network model.
    - data_iter: The data iterator.
    - device (torch.device, optional): The device on which to perform computations. Defaults to None.

    Returns:
    - float: The accuracy of the model on the dataset.
    
    '''
    if isinstance(net, nn.Module):
        net.eval() 
        if not device:
            device = next(iter(net.parameters())).device  
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            
            metric.add(d2l.accuracy(net(X), y), y.numel())      
    accuracy = metric[0] / metric[1]
    return accuracy
    

def evaluate_loss_gpu(net, data_iter, loss, device=None): 
    '''
    Evaluate the average loss of a neural network on a given dataset.

    Parameters:
    - net : The neural network model.
    - data_iter : The data iterator.
    - loss : The loss funtion.
    - device : The device on which to perform computations.

    Returns:
    - float: The average loss.
    '''
    num_batches = len(data_iter)
      
    if isinstance(net, nn.Module):
        net.eval() 
        if not device:
            device = next(iter(net.parameters())).device
 
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(loss(net(X), y), y.numel())   
    return metric[0] / num_batches



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
    - device : The device on which to perform computations.

    Returns:
    - net : The trained CNN model.
    - Trn_loss : The loss on training data iterator.
    - Val_loss : The loss on validation data iterator.
    
    '''
    
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    

    # Give different weights to different classes in the loss function to 
    # make the model pay more attention to the 'one' class.
    weights = torch.tensor([0.20, 0.80], device=device) 
    loss = nn.CrossEntropyLoss(weight=weights)  
    optimizer = torch.optim.Adadelta(net.parameters())

    timer = d2l.Timer()
    
    Trn_loss = np.empty(0)
    Val_loss = np.empty(0)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        metric = d2l.Accumulator(2)
        net.train()

        for i, (X, y) in enumerate(train_iter):  
            timer.start()
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()   
            with torch.no_grad():
                metric.add(l * X.shape[0], X.shape[0])
                
            timer.stop()
        Trn_loss = np.append(Trn_loss, metric[0] / metric[1])
        
             

        val_loss = evaluate_loss_gpu(net, val_iter,loss)
        Val_loss = np.append(Val_loss, val_loss)
       
       

        print(f'trn_loss {l:.3f},val_loss{val_loss:.3f}')
 
        
    return net, Trn_loss, Val_loss
        















"""
# =============================================================================
# #=================01 Read Data(training & validation datasets)===============
# =============================================================================
"""

depth = XX   # (m)
Batch_size = 256


SSTA = xr.open_dataset('./SSTA.nc').SSTA.values.squeeze()
SSHA = xr.open_dataset('./SSHA.nc').SSHA.values.squeeze()
SMHW = xr.open_dataset('./SMHW.nc').SMHW.values.squeeze()

lon1 = xr.open_dataset('./SSTA.nc').lon.values.squeeze()
lat1 = xr.open_dataset('./SSTA.nc').lat.values.squeeze()
time1 = xr.open_dataset('./SSTA.nc').time.values.squeeze()
lon2 = xr.open_dataset('./SSHA.nc').lon.values.squeeze()
lat2 = xr.open_dataset('./SSHA.nc').lat.values.squeeze()
time2 = xr.open_dataset('./SSHA.nc').time.values.squeeze()
lon3 = xr.open_dataset('./SMHW.nc').lon.values.squeeze()
lat3 = xr.open_dataset('./SMHW.nc').lat.values.squeeze()
time3 = xr.open_dataset('./SMHW.nc').time.values.squeeze()

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
YY_TRN = np.nan_to_num(SMHW[trn_index,:,:])

XX_TRN = torch.tensor(XX_TRN, dtype=torch.float32)
YY_TRN = torch.tensor(YY_TRN, dtype=torch.long)


train_ids = TensorDataset(XX_TRN, YY_TRN) 
trn_iter = DataLoader(dataset=train_ids, batch_size=Batch_size, shuffle=True)



#-------------------------------For validation datasets-------------------------------
SSHA2 = SSHA[val_index,:,:]
SSTA2  = SSTA[val_index,:,:]
SSHA2 = standardization(SSHA2,SSHA2)
SSTA2 = standardization(SSTA2,SSTA2)

XX_VAL = np.stack((SSTA2,SSHA2),axis=1)    
YY_VAL = SMHW[val_index,:,:]

XX_VAL = np.nan_to_num(XX_VAL)
YY_VAL = np.nan_to_num(YY_VAL)

XX_VAL = torch.tensor(XX_VAL, dtype=torch.float32)
YY_VAL = torch.tensor(YY_VAL, dtype=torch.long)

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
    nn.Conv2d(2, 16, kernel_size=3,padding=1), nn.Sigmoid(),
    nn.Conv2d(16, 32, kernel_size=3,padding=1), nn.Sigmoid(),
    nn.Conv2d(32, 2, kernel_size=3,padding=1))


X = torch.rand(size=(1, channel, len(LAT), len(LON)), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)


num_epochs = 100

net, Trn_mse, Val_mse = train_CNN_model(net, trn_iter, val_iter, num_epochs, 'sequential_model_'+str(depth)+'.pth')


print('--------------------------------------------------------')
scio.savemat('./Output/Trn_mse'+str(depth)+'.mat',{'Trn_mse':Trn_mse})  
scio.savemat('./Output/Val_mse'+str(depth)+'.mat',{'Val_mse':Val_mse})  

