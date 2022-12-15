import time
import torch

calculate_loss_over_all_values = False

def get_batch(source, i, batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    observation = torch.stack(
        torch.stack([item[0] for item in data]).chunk(input_window, 1))  
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return observation, target

def get_batch_mul(source, i, batch_size, input_window, number_of_cols):  
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    observation = torch.stack(
        torch.stack([item[0] for item in data]).chunk(input_window*number_of_cols, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window*number_of_cols, 1))
    
    return observation, target

def train(model, optimizer, criterion, scheduler, clip_grad_norm, train_data, batch_size, epoch, epochs, input_window, output_window):
    model.train()  
    total_loss = 0.0
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size, input_window)
        
        output = model(data)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm, norm_type=2.0)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 2)  # / 5
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d}/{} | {:5d}/{:5d} batches | '
                  'lr {:02.10f} | {:6.2f} ms | '
                  'train loss {:5.5f}'.format(
                epoch, epochs, batch, len(train_data) // batch_size, scheduler.get_last_lr()[0],
                                      elapsed * 1000 / log_interval,
                cur_loss))
            total_loss = 0.0
            start_time = time.time()

def train_mul(model, optimizer, criterion, scheduler, clip_grad_norm, train_data, batch_size, epoch, epochs, input_window, output_window, number_of_cols, device):
    model.train()  
    total_loss = 0.0
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch_mul(train_data, i, batch_size, input_window, number_of_cols)
        
        output = model(data)

        targets_temp = torch.zeros([output_window,1,1],device=device)
        for j in reversed(range(output_window)):            
            targets_temp[-j+output_window-1] = targets[-j*number_of_cols -1]    
        
        if calculate_loss_over_all_values:
            loss = criterion(output, targets_temp)
        else:
            loss = criterion(output[-output_window:], targets_temp[-output_window:])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm, norm_type=2.0)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 2)  # / 5
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d}/{} | {:5d}/{:5d} batches | '
                  'lr {:02.10f} | {:6.2f} ms | '
                  'train loss {:5.5f}'.format(
                epoch, epochs, batch, len(train_data) // batch_size, scheduler.get_last_lr()[0],
                                      elapsed * 1000 / log_interval,
                cur_loss))
            total_loss = 0.0
            start_time = time.time()


def evaluate(eval_model, criterion, data_source, input_window, output_window):
    eval_model.eval()
    total_loss = 0.0
    eval_batch_size = 32
    with torch.inference_mode():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size, input_window)
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)
    
    
def evaluate_mul(eval_model, criterion, data_source, input_window, output_window, number_of_cols, device):
    eval_model.eval()
    total_loss = 0.0
    eval_batch_size = 1
    with torch.inference_mode():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch_mul(data_source, i, eval_batch_size, input_window, number_of_cols)             
            output = eval_model(data)
            
            targets_temp = torch.zeros([output_window,1,1],device=device)
            for j in reversed(range(output_window)):
                targets_temp[-j+output_window-1] = targets[-j*number_of_cols -1]
            
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets_temp).cpu().item()
            else:
                total_loss += len(data[0])* criterion(output[-output_window:], targets_temp[-output_window:]).cpu().item()
    return total_loss / len(data_source)  
 
