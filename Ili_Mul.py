import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import time
import math
from matplotlib import pyplot
import matplotlib.pyplot as plt
torch.manual_seed(0)
np.random.seed(0)
from models_v2 import *
from procedures import *
from data_loader import *

### Begin of hyperparameters setting
data = "Ili-AF"                # Options: "ETT", "Electricity", "weather" 
dataset = "national_illness.csv"       # Options: "ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ECL.csv", "WTH.csv" 
target = "OT"               # Options: "OT" for {ETTh1.csv, ETTh2.csv, ETTm1.csv}, "MT_320" for ECL.csv, "WetBulbCelsius" for weather
number_of_cols = 7          # Options: 7 for {ETTh1.csv, ETTh2.csv, ETTm1.csv}, 321 for ECL.csv, 12 for WTH.csv

input_window = 36           # input_window = previous time steps - output_window
output_window = 24
start_lr = 0.00001
epochs = 15
loss = "MAE"  # options MSE and MAE


nhidden=128
num_layers=1
nhead=8
dropout=0.1
max_len_positional_encoding = 2000

scheduler_gamma = 0.97
batch_size = 1
step_size = 2
clip_grad_norm = 0.25

model_summary = True
model_save = False
with_Unet = True

### End of hyperparameters setting




if loss == "MSE":
    criterion = nn.MSELoss()
else:
    criterion = nn.L1Loss()
    

print("Multivariate")
print("data: " + data)
print("dataset: " + dataset)
print("target: " + target)
print("input_window: " + str(input_window))
print("output_window: " + str(output_window))
#print("batch_size: " + str(batch_size))
print("start_lr: " + str(start_lr))
print("epochs: " + str(epochs))
print("withUNet: " + str(with_Unet))
print("Loss: " + loss)
print("===================================")






train_data, val_data, test_data = get_Ili_AF_mul(dataset=dataset, target=target, input_window=input_window, output_window=output_window, device=device, number_of_cols=number_of_cols)


if with_Unet == True:
    model = v5_Ili_Mul_UNet(nhidden=nhidden, num_layers=num_layers, nhead=nhead, dropout=dropout, in_channels=input_window, out_channels=output_window, max_len_positional_encoding=max_len_positional_encoding, number_of_cols=number_of_cols).to(device)
else:
    model = v5_Ili_Mul_noUNet(nhidden=nhidden, num_layers=num_layers, nhead=nhead, dropout=dropout, in_channels=input_window, out_channels=output_window, max_len_positional_encoding=max_len_positional_encoding, number_of_cols=number_of_cols).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=scheduler_gamma)




if model_summary == True:
    print("=== Begin of model summary")
    print(model)
    print("=== End of model summary")

    
if model_save == True:
   what_to_save = {
   'epoch': epochs,
   'model': model.state_dict(),
   'optimizer': optimizer.state_dict(),
   'scheduler': scheduler.state_dict()
   }
   torch.save(what_to_save, "saved_models/checkpoint-mul-" + str(with_Unet) + "-" + dataset + "-" + str(input_window) + "-" + str(output_window) + ".pth")
    




### Begin of training and evaluating
print("=== Begin of training and evaluating")
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_mul(model, optimizer, criterion, scheduler, clip_grad_norm, train_data, batch_size, epoch, epochs, input_window, output_window, number_of_cols, device)

    '''
    if (epoch % 5 == 0):
        test_loss = evaluate_mul(model, criterion, test_data, input_window, output_window, number_of_cols, device)
        print('-' * 50)
        print('| end of epoch {:3d}/{} | time: {:5.2f}s | test loss {:5.5f}'.format(epoch, epochs,(time.time() - epoch_start_time), test_loss))
        print('-' * 50)
    else:
        pass
        #val_loss = evaluate_mul(model, criterion, val_data, input_window, output_window, number_of_cols, device)
        #test_loss = evaluate_mul(model, criterion, test_data, input_window, output_window, number_of_cols, device)
    '''
    try:
        val_loss = evaluate_mul(model, criterion, val_data, input_window, output_window, number_of_cols, device)
        print('| end of epoch {:3d}/{} | time: {:5.2f}s | valid loss {:5.5f}'.format(epoch, epochs,(time.time() - epoch_start_time), val_loss))
    except ZeroDivisionError:
        pass 
        
    test_loss = evaluate_mul(model, criterion, test_data, input_window, output_window, number_of_cols, device)
    
    print('-' * 100)
    print('| end of epoch {:3d}/{} | time: {:5.2f}s | test loss {:5.5f}'.format(epoch, epochs,(time.time() - epoch_start_time), test_loss))
    print('-' * 100)

    
    scheduler.step()

print("final test_loss:")
print(evaluate_mul(model, criterion, test_data, input_window, output_window, number_of_cols, device))

### End of training and evaluating














