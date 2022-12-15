# pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
import copy
from typing import Optional, Any, Union, Callable
import torch.nn as nn
import math
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm


class TransformerEncoderLayer(nn.Module):
    
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 512, dropout: float = 0.2,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-4, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout, inplace=True)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout, inplace=True)
        self.dropout2 = Dropout(dropout, inplace=True)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)[0] #False
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        #x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.linear2(self.activation(self.linear1(x)))
        return self.dropout2(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len_positional_encoding):
        super(PositionalEncoding, self).__init__()
        max_len = max_len_positional_encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_len_positional_encoding) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]



class v5_Ili_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_Ili_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 96, 3, 1)
        self.conv4 = self.contract_block(96, 192, 3, 1)

        self.upconv4 = self.expand_block(192, 96, 3, 1)
        self.upconv3 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.05    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            #torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1), #
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.05, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.05, inplace=True)
        )
        return expand


class v5_Ili_Mul_UNet_ablation(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_Ili_Mul_UNet_ablation, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        
        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 192, 3, 1)
        
        self.upconv3 = self.expand_block(192, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        
        '''
        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 96, 3, 1)
        self.conv4 = self.contract_block(96, 192, 3, 1)

        self.upconv4 = self.expand_block(192, 96, 3, 1)
        self.upconv3 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        '''
        '''
        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 96, 3, 1)
        self.conv4 = self.contract_block(96, 192, 3, 1)
        self.conv5 = self.contract_block(192, 256, 3, 1)
        
        self.upconv5 = self.expand_block(256, 192, 3, 1)
        self.upconv4 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv3 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        '''

    def init_weights(self):
        initrange = 0.05    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))
        
        
        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        
        '''
        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        '''
        '''
        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        upconv5 = self.upconv5(conv5)
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        '''

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            #torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1), #
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.05, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.05, inplace=True)
        )
        return expand



class v5_Ili_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_Ili_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        

class v5_Exchange_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_Exchange_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 96, 3, 1)
        self.conv4 = self.contract_block(96, 192, 3, 1)

        self.upconv4 = self.expand_block(192, 96, 3, 1)
        self.upconv3 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1), #
            #torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.25, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.25, inplace=True)
        )
        return expand


class v5_Exchange_Mul_UNet_ablation(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_Exchange_Mul_UNet_ablation, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()
        
        '''
        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 192, 3, 1)
        

        
        self.upconv3 = self.expand_block(192, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        '''
        
        
        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 96, 3, 1)
        self.conv4 = self.contract_block(96, 192, 3, 1)

        self.upconv4 = self.expand_block(192, 96, 3, 1)
        self.upconv3 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        
        '''
        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 96, 3, 1)
        self.conv4 = self.contract_block(96, 192, 3, 1)
        self.conv5 = self.contract_block(192, 256, 3, 1)

        self.upconv5 = self.expand_block(256, 192, 3, 1)
        self.upconv4 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv3 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        '''

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))
        
        '''
        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        '''
        
        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        
        '''
        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        upconv5 = self.upconv5(conv5)
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        '''

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            #torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1), #
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.25, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.25, inplace=True)
        )
        return expand



class v5_Exchange_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_Exchange_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        

class v5_Electricity_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_Electricity_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=32, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 96, 3, 1)
        self.conv4 = self.contract_block(96, 192, 3, 1)

        self.upconv4 = self.expand_block(192, 96, 3, 1)
        self.upconv3 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.01    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            #torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1), #
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.4, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.4, inplace=True)
        )
        return expand


class v5_Electricity_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_Electricity_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
   
   

class v5_Weather_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_Weather_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=128, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 72, 3, 1)
        self.conv4 = self.contract_block(72, 128, 3, 1)

        self.upconv4 = self.expand_block(128, 72, 3, 1)
        self.upconv3 = self.expand_block(72 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            #torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1), #
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.4, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.4, inplace=True)
        )
        return expand


class v5_Weather_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_Weather_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=128, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask




class v5_ETTm2_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_ETTm2_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=1024, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()


        self.conv1 = self.contract_block(self.in_channels, 30, 3, 1)
        self.conv2 = self.contract_block(30, 60, 3, 1)
        self.conv3 = self.contract_block(60, 120, 3, 1)
        
        self.upconv3 = self.expand_block(120, 60, 3, 1)
        self.upconv2 = self.expand_block(60 * 2, 30, 3, 1)
        self.upconv1 = self.expand_block(30 * 2, self.out_channels, 3, 1)
        '''
        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 60, 3, 1)
        self.conv3 = self.contract_block(60, 120, 3, 1)
        self.conv4 = self.contract_block(120, 180, 3, 1)

        self.upconv4 = self.expand_block(180, 120, 3, 1)
        self.upconv3 = self.expand_block(120 * 2, 60, 3, 1)
        self.upconv2 = self.expand_block(60 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        '''
        

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        
        '''
        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        '''

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.3, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            #torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.3, inplace=True)
        )
        return expand



        

class v5_ETTm2_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_ETTm2_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class v5_ETTh1_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_ETTh1_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 72, 3, 1)
        self.conv4 = self.contract_block(72, 96, 3, 1)

        self.upconv4 = self.expand_block(96, 72, 3, 1)
        self.upconv3 = self.expand_block(72 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.01    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.3, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.3, inplace=True)
        )
        return expand


class v5_ETTh1_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_ETTh1_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        

class v5_ETTm1_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_ETTm1_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class v5_ETTm1_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_ETTm1_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=128, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 72, 3, 1)
        self.conv4 = self.contract_block(72, 96, 3, 1)

        self.upconv4 = self.expand_block(96, 72, 3, 1)
        self.upconv3 = self.expand_block(72 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.01    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.3, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.3, inplace=True)
        )
        return expand



class v5_ETTh2_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_ETTh2_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=1024, layer_norm_eps=1e-5)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 60, 3, 1)
        self.conv3 = self.contract_block(60, 120, 3, 1)
        self.conv4 = self.contract_block(120, 180, 3, 1)

        self.upconv4 = self.expand_block(180, 120, 3, 1)
        self.upconv3 = self.expand_block(120 * 2, 60, 3, 1)
        self.upconv2 = self.expand_block(60 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.15    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.05, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.05, inplace=True)
        )
        return expand


class v5_ETTh2_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_ETTh2_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



class v5_ETTm2_Uni_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels,max_len_positional_encoding): #nhidden=1024, num_layers=1, nhead=8, dropout=0.5, in_channels=input_window, out_channels=output_window
        super(v5_ETTm2_Uni_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        #self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 60, 3, 1)
        self.conv3 = self.contract_block(60, 120, 3, 1)
        self.conv4 = self.contract_block(120, 180, 3, 1)

        self.upconv4 = self.expand_block(180, 120, 3, 1)
        self.upconv3 = self.expand_block(120 * 2, 60, 3, 1)
        self.upconv2 = self.expand_block(60 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block
        
        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1), #MaxPool1d
            torch.nn.Dropout(p=0.25, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
        torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.25, inplace=True)
        )
        return expand 
        

'''
class v5_ETTm2_Uni_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding): 
        super(v5_ETTm2_Uni_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels# * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        #self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 72, 3, 1)
        self.conv4 = self.contract_block(72, 96, 3, 1)

        self.upconv4 = self.expand_block(96, 72, 3, 1)
        self.upconv3 = self.expand_block(72 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.01    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.3, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.3, inplace=True)
        )
        return expand
'''


class v5_ETTm2_Uni_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding): 
        super(v5_ETTm2_Uni_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels# * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        #self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask




#####################################################################

class Transformer_ETTh1_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels,max_len_positional_encoding): #nhidden=1024, num_layers=1, nhead=8, dropout=0.5, in_channels=input_window, out_channels=output_window
        super(Transformer_ETTh1_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(in_channels, 48, 3, 1)
        self.conv2 = self.contract_block(48, 96, 3, 1)
        self.conv3 = self.contract_block(96, 192, 3, 1)
        self.conv4 = self.contract_block(192, 384, 3, 1)

        self.upconv4 = self.expand_block(384, 192, 3, 1)
        self.upconv3 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv2 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv1 = self.expand_block(48 * 2, out_channels, 3, 1)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.5)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.5)
        )
        return expand
        
        
class Transformer_ETTh1_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels,max_len_positional_encoding):
        super(Transformer_ETTh1_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)  
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class Transformer_ETTh1_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_ETTh1_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=128, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 48, 3, 1)
        self.conv2 = self.contract_block(48, 96, 3, 1)
        self.conv3 = self.contract_block(96, 192, 3, 1)
        self.conv4 = self.contract_block(192, 384, 3, 1)

        self.upconv4 = self.expand_block(384, 192, 3, 1)
        self.upconv3 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv2 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv1 = self.expand_block(48 * 2, self.out_channels, 3, 1)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.1)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.1)
        )
        return expand


class Transformer_ETTh1_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_ETTh1_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=128, norm_first=True, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

        
  
class Transformer_ETTh2_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels,max_len_positional_encoding): #nhidden=1024, num_layers=1, nhead=8, dropout=0.5, in_channels=input_window, out_channels=output_window
        super(Transformer_ETTh2_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(in_channels, 48, 3, 1)
        self.conv2 = self.contract_block(48, 96, 3, 1)
        self.conv3 = self.contract_block(96, 192, 3, 1)
        self.conv4 = self.contract_block(192, 384, 3, 1)

        self.upconv4 = self.expand_block(384, 192, 3, 1)
        self.upconv3 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv2 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv1 = self.expand_block(48 * 2, out_channels, 3, 1)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.5)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.5)
        )
        return expand

        
class Transformer_ETTh2_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels,max_len_positional_encoding): #nhidden=1024, num_layers=1, nhead=8, dropout=0.5, in_channels=input_window, out_channels=output_window
        super(Transformer_ETTh2_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



class Transformer_ETTh2_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_ETTh2_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 48, 3, 1)
        self.conv2 = self.contract_block(48, 96, 3, 1)
        self.conv3 = self.contract_block(96, 192, 3, 1)
        self.conv4 = self.contract_block(192, 384, 3, 1)

        self.upconv4 = self.expand_block(384, 192, 3, 1)
        self.upconv3 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv2 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv1 = self.expand_block(48 * 2, self.out_channels, 3, 1)

    def init_weights(self):
        initrange = 0.01
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.5)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.5)
        )
        return expand


class Transformer_ETTh2_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_ETTh2_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


        
class Transformer_ETTm1_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels,max_len_positional_encoding): #nhidden=1024, num_layers=1, nhead=8, dropout=0.5, in_channels=input_window, out_channels=output_window
        super(Transformer_ETTm1_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(in_channels, 48, 3, 1)
        self.conv2 = self.contract_block(48, 96, 3, 1)
        self.conv3 = self.contract_block(96, 192, 3, 1)
        self.conv4 = self.contract_block(192, 384, 3, 1)

        self.upconv4 = self.expand_block(384, 192, 3, 1)
        self.upconv3 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv2 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv1 = self.expand_block(48 * 2, out_channels, 3, 1)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.5)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.5)
        )
        return expand 
        

class Transformer_ETTm1_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels,max_len_positional_encoding): #nhidden=1024, num_layers=1, nhead=8, dropout=0.5, in_channels=input_window, out_channels=output_window
        super(Transformer_ETTm1_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
 
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class Transformer_ETTm1_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_ETTm1_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 48, 3, 1)
        self.conv2 = self.contract_block(48, 96, 3, 1)
        self.conv3 = self.contract_block(96, 192, 3, 1)
        self.conv4 = self.contract_block(192, 384, 3, 1)

        self.upconv4 = self.expand_block(384, 192, 3, 1)
        self.upconv3 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv2 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv1 = self.expand_block(48 * 2, self.out_channels, 3, 1)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.5)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.5)
        )
        return expand


class Transformer_ETTm1_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_ETTm1_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class Transformer_ETTm2_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels,max_len_positional_encoding): #nhidden=1024, num_layers=1, nhead=8, dropout=0.5, in_channels=input_window, out_channels=output_window
        super(Transformer_ETTm2_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(in_channels, 48, 3, 1)
        self.conv2 = self.contract_block(48, 96, 3, 1)
        self.conv3 = self.contract_block(96, 192, 3, 1)
        self.conv4 = self.contract_block(192, 384, 3, 1)

        self.upconv4 = self.expand_block(384, 192, 3, 1)
        self.upconv3 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv2 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv1 = self.expand_block(48 * 2, out_channels, 3, 1)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.5)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.5)
        )
        return expand 
        


class Transformer_ETTm2_n_hits_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_ETTm2_n_hits_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 96, 3, 1)
        self.conv4 = self.contract_block(96, 192, 3, 1)

        self.upconv4 = self.expand_block(192, 96, 3, 1)
        self.upconv3 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.5, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.5, inplace=True)
        )
        return expand



        

class Transformer_ETTm2_n_hits_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_ETTm2_n_hits_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



        
class Transformer_ECL_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels,max_len_positional_encoding): #nhidden=1024, num_layers=1, nhead=8, dropout=0.5, in_channels=input_window, out_channels=output_window
        super(Transformer_ECL_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(in_channels, 48, 3, 1)
        self.conv2 = self.contract_block(48, 96, 3, 1)
        self.conv3 = self.contract_block(96, 192, 3, 1)
        self.conv4 = self.contract_block(192, 384, 3, 1)

        self.upconv4 = self.expand_block(384, 192, 3, 1)
        self.upconv3 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv2 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv1 = self.expand_block(48 * 2, out_channels, 3, 1)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.1)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.1)
        )
        return expand         


class Transformer_ECL_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels,max_len_positional_encoding): #nhidden=1024, num_layers=1, nhead=8, dropout=0.5, in_channels=input_window, out_channels=output_window
        super(Transformer_ECL_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class Transformer_ECL_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_ECL_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 48, 3, 1)
        self.conv2 = self.contract_block(48, 96, 3, 1)
        self.conv3 = self.contract_block(96, 192, 3, 1)
        self.conv4 = self.contract_block(192, 384, 3, 1)

        self.upconv4 = self.expand_block(384, 192, 3, 1)
        self.upconv3 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv2 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv1 = self.expand_block(48 * 2, self.out_channels, 3, 1)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.1)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.1)
        )
        return expand


class Transformer_ECL_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_ECL_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class v5_ECL_n_hits_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_ECL_n_hits_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=128, layer_norm_eps=1e-5)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 96, 3, 1)
        self.conv4 = self.contract_block(96, 128, 3, 1)

        self.upconv4 = self.expand_block(128, 96, 3, 1)
        self.upconv3 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.01    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.4, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.4, inplace=True)
        )
        return expand


class v5_ECL_n_hits_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_ECL_n_hits_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=256, layer_norm_eps=1e-4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



class v5_ETTm2_n_hits_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_ETTm2_n_hits_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=512, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 96, 3, 1)
        self.conv4 = self.contract_block(96, 256, 3, 1)

        self.upconv4 = self.expand_block(256, 96, 3, 1)
        self.upconv3 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.01    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            #torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.5, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ReLU(),
            #torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            #torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.5, inplace=True)
        )
        return expand


class v5_ETTm2_n_hits_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(v5_ETTm2_n_hits_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=512, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    
class Transformer_Weather_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels,max_len_positional_encoding): #nhidden=1024, num_layers=1, nhead=8, dropout=0.5, in_channels=input_window, out_channels=output_window
        super(Transformer_Weather_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=64, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(in_channels, 48, 3, 1)
        self.conv2 = self.contract_block(48, 96, 3, 1)
        self.conv3 = self.contract_block(96, 192, 3, 1)
        self.conv4 = self.contract_block(192, 512, 3, 1)

        self.upconv4 = self.expand_block(512, 192, 3, 1)
        self.upconv3 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv2 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv1 = self.expand_block(48 * 2, out_channels, 3, 1)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.1)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.1)
        )
        return expand      
        
        
class Transformer_Weather_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels,max_len_positional_encoding): #nhidden=1024, num_layers=1, nhead=8, dropout=0.5, in_channels=input_window, out_channels=output_window
        super(Transformer_Weather_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=64, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
   

class Transformer_Weather_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_Weather_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=64, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 48, 3, 1)
        self.conv2 = self.contract_block(48, 96, 3, 1)
        self.conv3 = self.contract_block(96, 192, 3, 1)
        self.conv4 = self.contract_block(192, 512, 3, 1)

        self.upconv4 = self.expand_block(512, 192, 3, 1)
        self.upconv3 = self.expand_block(192 * 2, 96, 3, 1)
        self.upconv2 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv1 = self.expand_block(48 * 2, self.out_channels, 3, 1)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.1)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.1)
        )
        return expand


class Transformer_Weather_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_Weather_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=64, layer_norm_eps=1e-6)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask   
        
          

class Transformer_Exchange_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_Exchange_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=128, layer_norm_eps=1e-4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 96, 3, 1)
        self.conv4 = self.contract_block(96, 192, 3, 1)

        self.upconv4 = self.expand_block(192, 96, 3, 1)
        self.upconv3 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.3, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.3, inplace=True)
        )
        return expand

       
class Transformer_Exchange_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_Exchange_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=128, layer_norm_eps=1e-4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask        


class Transformer_ILI_Mul_UNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_ILI_Mul_UNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=128, layer_norm_eps=1e-4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

        self.conv1 = self.contract_block(self.in_channels, 24, 3, 1)
        self.conv2 = self.contract_block(24, 48, 3, 1)
        self.conv3 = self.contract_block(48, 96, 3, 1)
        self.conv4 = self.contract_block(96, 192, 3, 1)

        self.upconv4 = self.expand_block(192, 96, 3, 1)
        self.upconv3 = self.expand_block(96 * 2, 48, 3, 1)
        self.upconv2 = self.expand_block(48 * 2, 24, 3, 1)
        self.upconv1 = self.expand_block(24 * 2, self.out_channels, 3, 1)
        

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        ### Begin of the U-net block
        output = torch.reshape(output, (output.shape[1], output.shape[0], output.shape[2]))

        conv1 = self.conv1(output)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconv1 = torch.reshape(upconv1, (upconv1.shape[1], upconv1.shape[0], upconv1.shape[2]))
        ### End of the U-net block

        output = self.decoder(upconv1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.3, inplace=True)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Dropout(p=0.3, inplace=True)
        )
        return expand



        

class Transformer_ILI_Mul_noUNet(nn.Module):
    def __init__(self, nhidden, num_layers, nhead, dropout, in_channels, out_channels, max_len_positional_encoding, number_of_cols): 
        super(Transformer_ILI_Mul_noUNet, self).__init__()
        self.model_type = 'Transformer for Time Series Forecasting'
        self.nhidden = nhidden
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.in_channels = in_channels * number_of_cols
        self.out_channels = out_channels
        self.max_len_positional_encoding=max_len_positional_encoding
        self.number_of_cols = number_of_cols
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nhidden,max_len_positional_encoding)

        self.encoder_layer = TransformerEncoderLayer(d_model=nhidden, nhead=nhead, dropout=dropout, dim_feedforward=128, layer_norm_eps=1e-4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
     
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, ss):
        mask = (torch.triu(torch.ones(ss, ss)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask