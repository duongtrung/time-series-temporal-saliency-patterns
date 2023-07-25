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
data = "ETT"                # Options: "ETT", "Electricity", "weather" 
dataset = "ETTh1.csv"       # Options: "ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ECL.csv", "WTH.csv" 
target = "OT"               # Options: "OT" for {ETTh1.csv, ETTh2.csv, ETTm1.csv}, "MT_320" for ECL.csv, "WetBulbCelsius" for weather

input_window = 48           # input_window = previous time steps - output_window
output_window = 24
start_lr = 0.00001
epochs = 5
loss = "MSE"  # options MSE and MAE


nhidden=1024
num_layers=2
nhead=8
dropout=0.5
max_len_positional_encoding = 5000

scheduler_gamma = 0.87
batch_size = 32
step_size = 2
clip_grad_norm = 0.5

model_summary = True
model_save = True
with_Unet = False

### End of hyperparameters setting




if loss == "MSE":
    criterion = nn.MSELoss()
else:
    criterion = nn.L1Loss()
    

print("===================================")
print("Univariate")
print("data: " + data)
print("dataset: " + dataset)
print("target: " + target)
print("input_window: " + str(input_window))
print("output_window: " + str(output_window))
print("batch_size: " + str(batch_size))
print("start_lr: " + str(start_lr))
print("epochs: " + str(epochs))
print("withUNet: " + str(with_Unet))
print("Loss: " + loss)
print("===================================")



train_data, val_data, test_data = get_ETTh1_data(dataset=dataset, target=target, input_window=input_window, output_window=output_window, device=device)

if with_Unet == True:
    model = Transformer_ETTh1_UNet(nhidden=nhidden, num_layers=num_layers, nhead=nhead, dropout=dropout, in_channels=input_window, out_channels=output_window, max_len_positional_encoding=max_len_positional_encoding).to(device)
else:
    model = Transformer_ETTh1_noUNet(nhidden=nhidden, num_layers=num_layers, nhead=nhead, dropout=dropout, in_channels=input_window, out_channels=output_window, max_len_positional_encoding=max_len_positional_encoding).to(device)

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
   torch.save(what_to_save, "saved_models/checkpoint-uni-" + str(with_Unet) + "-" + dataset + "-" + str(input_window) + "-" + str(output_window) + ".pth")
    




### Begin of training and evaluating
print("=== Begin of training and evaluating")
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model, optimizer, criterion, scheduler, clip_grad_norm, train_data, batch_size, epoch, epochs, input_window, output_window)

    if (epoch % 5 == 0):
        pass
    else:
        val_loss = evaluate(model, criterion, test_data, input_window, output_window)

    print('-' * 100)

    print('| end of epoch {:3d}/{} | time: {:5.2f}s | valid loss {:5.5f}'.format(epoch, epochs,
                                                                                 (time.time() - epoch_start_time),
                                                                                 val_loss))
    print('-' * 100)
    scheduler.step()

print("evaluate test_data:")
print(evaluate(model, criterion, test_data, input_window, output_window))

### End of training and evaluating














