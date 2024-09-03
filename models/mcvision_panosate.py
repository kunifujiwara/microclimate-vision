import logging
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import os
from datasets.mcvision_dataset import MCVisionDataset

from .modelutils import get_train_mean_std, get_val_rmse_baseline, get_optimizer, get_scheduler, get_log_loss_metrics, log_loss_average

############
class MCVisionNet_panosate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if not self.args.lstm_zero_init:
            self.SAT = True
        else:
            self.SAT = False        
        
        if args.cnn_architecture == "resnet50":
            self.pcnn1 = models.resnet50()
            self.scnn1 = models.resnet50()
            coef = 4
        elif args.cnn_architecture == "resnet18":
            self.pcnn1 = models.resnet18()
            self.scnn1 = models.resnet18()
            coef = 1

        self.coef = coef * (int(args.pano_x/128))**2
        self.pcnn1.fc = nn.Identity()
        self.pcnn1.avgpool = nn.Identity()
        self.scnn1.fc = nn.Identity()
        self.scnn1.avgpool = nn.Identity()

        if self.args.cnn_subtract: 
            lin_proj_in_dim = 4096 * 2 * self.coef
        else:             
            lin_proj_in_dim = 4096 * 2 * self.coef + 4096 * self.coef
            if args.cnn_architecture == "resnet50":
                self.pcnn2 = models.resnet50()
                self.scnn2 = models.resnet50()
            elif args.cnn_architecture == "resnet18":
                self.pcnn2 = models.resnet18()
                self.scnn2 = models.resnet18()
            self.pcnn2.fc = nn.Identity()
            self.pcnn2.avgpool = nn.Identity()
            self.scnn2.fc = nn.Identity()
            self.scnn2.avgpool = nn.Identity()

        self.lin_proj = nn.Linear(lin_proj_in_dim, self.args.cnn_feature_dim * 2) #nn.Linear(4096, 128)
        self.ln1 = nn.LayerNorm(self.args.cnn_feature_dim * 2)
        
        self.lstm = nn.LSTM(
            input_size = self.args.lstm_input_size,
            hidden_size = self.args.lstm_hidden_units,
            batch_first = True,
            num_layers = self.args.lstm_num_layers,
            dropout=self.args.lstm_dropout_ratio
        )
        self.ln2 = nn.LayerNorm(self.args.lstm_hidden_units)

        self.regressor = nn.Sequential(
            nn.Linear(self.args.cnn_feature_dim * 2 + self.args.lstm_hidden_units, 64),
            nn.ReLU(),
            nn.Dropout(self.args.regressor_dropout_ratio),
            nn.Linear(64, 1)
        )

        self.init_h = nn.Linear(512*self.coef, self.args.lstm_hidden_units)
        self.init_c = nn.Linear(512*self.coef, self.args.lstm_hidden_units)

        self.criterion = torch.nn.MSELoss() 

    def initHidden(self, batch_size, penc_im=None, senc_im=None):
        if self.args.lstm_zero_init:
            return (torch.zeros(self.args.lstm_num_layers, batch_size, self.args.lstm_hidden_units).cuda(),
                    torch.zeros(self.args.lstm_num_layers, batch_size, self.args.lstm_hidden_units).cuda())
        elif self.SAT:
            # Use the same initialization as found in Show,Attend and Tell paper
            enc_im = (penc_im.mean(dim=1) + senc_im.mean(dim=1))/2
            h = self.init_h(enc_im).unsqueeze(0)
            c = self.init_c(enc_im).unsqueeze(0)

            # Ensure that h,c scales with num of stacked layers
            h = h.repeat(self.args.lstm_num_layers, 1, 1)
            c = c.repeat(self.args.lstm_num_layers, 1, 1)

            return (h, c)

    def forward(self, pano1, pano2, sate1, sate2, numerical):
        batch_size = numerical.shape[0]
        
        px1 = self.pcnn1(pano1)        
        sx1 = self.scnn1(sate1)
        if self.args.cnn_subtract: 
            px2 = self.pcnn1(pano2) 
            sx2 = self.scnn1(sate2)
        else:
            px2 = self.pcnn2(pano2) 
            sx2 = self.scnn2(sate2)

        px1 = px1.reshape(batch_size, self.args.lstm_ft_map_size, 512*self.coef)
        px2 = px2.reshape(batch_size, self.args.lstm_ft_map_size, 512*self.coef)
        sx1 = sx1.reshape(batch_size, self.args.lstm_ft_map_size, 256*self.coef)
        sx2 = sx2.reshape(batch_size, self.args.lstm_ft_map_size, 256*self.coef)

        # initialize with the average of both panos
        (h0_x1,c0_x1) = self.initHidden(batch_size=batch_size, penc_im=px1, senc_im=sx1)###############################################################################
        (h0_x2,c0_x2) = self.initHidden(batch_size=batch_size, penc_im=px2, senc_im=sx2)###############################################################################
        
        h0,c0 = (h0_x1+h0_x2),(c0_x1+c0_x2)
        # h0 = torch.zeros(num_layers, batch_size, hidden_units).requires_grad_().to(device)
        # c0 = torch.zeros(num_layers, batch_size, hidden_units).requires_grad_().to(device)
        _, (hn, _) = self.lstm(numerical, (h0, c0)) 
        # Only uses the final hidden state of the LSTM (basically not using all the other part of the sequence
        if self.args.lstm_layernorm:
            x3 = self.ln2(hn[0])
        else:
            x3 = hn[0]
        # x3 = self.mlp(hn)

        if self.args.cnn_subtract: 
            x_im_combined = torch.concat([(px2-px1),(sx2-sx1)],dim=2).view(batch_size,-1)
        else:
            x_im_combined = torch.concat([px1, px2, sx1, sx2],dim=2).view(batch_size,-1)
        x_im_combined = self.lin_proj(x_im_combined)
        if self.args.cnn_layernorm:
            x_im_combined = self.ln1(x_im_combined)

        # Maybe adding attention here? Which part of the panos should be focused on?        
        x = torch.cat((x_im_combined, x3), dim=1)
        x = self.regressor(x).view(-1)
        return x
