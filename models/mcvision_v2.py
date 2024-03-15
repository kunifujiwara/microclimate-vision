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
import pytorch_lightning as pl
from datasets.mcvision_pano_dataset import MCVisionPanoDataset

from .modelutils import get_train_mean_std

############
class MCVisionNet_v2(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.mean_t, self.std_t = get_train_mean_std(self.args)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.validation_step_outputs_rmse = []

        if not self.args.lstm_zero_init:
            self.SAT = True
        else:
            self.SAT = False
        
        
        self.cnn1 = models.resnet18()
        self.cnn1.fc = nn.Identity()
        self.cnn1.avgpool = nn.Identity()
        
        # self.cnn.fc = nn.Sequential(
        #     nn.Linear(self.cnn.fc.in_features, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU()
        # )

        self.cnn2 = models.resnet18()
        self.cnn2.fc = nn.Identity()
        self.cnn2.avgpool = nn.Identity()
        self.lin_proj = nn.Linear(4096*2,64)
        self.ln1 = nn.LayerNorm(64)
        
        self.lstm = nn.LSTM(input_size = self.args.lstm_input_size,
                            hidden_size = self.args.lstm_hidden_units,
                            batch_first = True,
                            num_layers = self.args.lstm_num_layers,
                            dropout=0.2
        )
        self.ln2 = nn.LayerNorm(self.args.lstm_hidden_units)
        self.mlp = nn.Sequential(
            nn.Linear(self.args.lstm_hidden_units, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.regressor = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        self.init_h = nn.Linear(512, self.args.lstm_hidden_units)
        self.init_c = nn.Linear(512, self.args.lstm_hidden_units)

        self.criterion = torch.nn.MSELoss() 

    def initHidden(self, batch_size, enc_im=None):
        if self.args.lstm_zero_init:
            return (torch.zeros(self.args.lstm_num_layers, batch_size, self.args.lstm_hidden_units).cuda(),
                    torch.zeros(self.args.lstm_num_layers, batch_size, self.args.lstm_hidden_units).cuda())
        elif self.SAT:
            # Use the same initialization as found in Show,Attend and Tell paper
            enc_im = enc_im.mean(dim=1)
            h = self.init_h(enc_im).unsqueeze(0)
            c = self.init_c(enc_im).unsqueeze(0)

            # Ensure that h,c scales with num of stacked layers
            h = h.repeat(self.args.lstm_num_layers, 1, 1)
            c = c.repeat(self.args.lstm_num_layers, 1, 1)

            return (h, c)

    def forward(self, image1, image2, numerical):
        batch_size = numerical.shape[0]
        
        x1 = self.cnn1(image1)
        x1 = x1.reshape(batch_size, self.args.lstm_ft_map_size,512)
        x2 = self.cnn2(image2)
        x2 = x2.reshape(batch_size, self.args.lstm_ft_map_size,512)
        
        # initialize with the average of both images
        (h0_x1,c0_x1) = self.initHidden(batch_size=batch_size, enc_im=x1)
        (h0_x2,c0_x2) = self.initHidden(batch_size=batch_size, enc_im=x2)
        
        h0,c0 = (h0_x1+h0_x2),(c0_x1+c0_x2)
        # h0 = torch.zeros(num_layers, batch_size, hidden_units).requires_grad_().to(device)
        # c0 = torch.zeros(num_layers, batch_size, hidden_units).requires_grad_().to(device)
        _, (hn, _) = self.lstm(numerical, (h0, c0)) 
        # Only uses the final hidden state of the LSTM (basically not using all the other part of the sequence
        hn = self.ln2(hn[0])
        x3 = hn
        # x3 = self.mlp(hn)

        x_im_combined = torch.concat([x1,x2],dim=2).view(batch_size,-1)
        x_im_combined = self.lin_proj(x_im_combined)
        x_im_combined = self.ln1(x_im_combined)

        # Maybe adding attention here? Which part of the images should be focused on?
        
        x = torch.cat((x_im_combined, x3), dim=1)
        x = self.regressor(x).view(-1)
        return x


    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr_initial,
            weight_decay=self.args.weight_decay
        )

        # Define the learning rate scheduler
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',  # Assuming you want to minimize a metric
                patience=self.args.lr_patience,
                factor=self.args.lr_factor,
                verbose=True  # If you want to log the LR reduction
            ),
            'name': 'lr_scheduler',  # Optional: Naming the scheduler
            'monitor': 'val_loss',  # Replace with your actual metric
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True  # Specific to ReduceLROnPlateau
        }

        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dset = MCVisionPanoDataset(args=self.args, process_type="train")

        # mean_std = train_dset.get_mean_std_for_t_scaling()
        # self.mean_t = mean_std["mean_t"]
        # self.std_t = mean_std["std_t"]

        return DataLoader(train_dset, 
                          batch_size=self.args.train_batch_size, 
                          num_workers=self.args.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          shuffle=False) 

    def val_dataloader(self):
        val_dset = MCVisionPanoDataset(args=self.args, process_type=self.args.val_type)

        return DataLoader(val_dset, 
                          batch_size=self.args.val_batch_size, 
                          num_workers=self.args.num_workers,
                          pin_memory=True, 
                          persistent_workers=True,
                          shuffle=False)
    

    def training_step(self, train_batch, batch_idx):
        images1 = train_batch["images1"]
        images2 = train_batch["images2"]
        numericals = train_batch["numericals"]
        targets = train_batch["targets"]
        outputs = self.forward(images1, images2, numericals)
        loss = self.criterion(outputs, targets)

        self.log('train_loss', loss)
        self.logger.experiment.add_scalar('train_loss', loss, self.global_step)

        self.training_step_outputs.append(loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        images1 = val_batch["images1"]
        images2 = val_batch["images2"]
        numericals = val_batch["numericals"]
        targets = val_batch["targets"]
        outputs = self.forward(images1, images2, numericals)
        loss = self.criterion(outputs, targets)

        if self.args.output_scaling:
            rmse_inv = self.evaluate_inverse_scaled_rmse(outputs, targets, self.mean_t, self.std_t)   
        else:
            rmse_inv = self.evaluate_rmse(outputs, targets)

        metrics = {
            "val_loss": loss,
            "val_RMSE": rmse_inv,
        }

        # self.log('val_loss', loss)
        # self.log('val_RMSE', rmse_inv)
        self.log_dict(metrics)

        self.logger.experiment.add_scalar('val_loss', loss, self.global_step)   
        self.logger.experiment.add_scalar('val_RMSE', rmse_inv, self.global_step)   

        self.validation_step_outputs.append(loss)
        self.validation_step_outputs_rmse.append(rmse_inv)

        return metrics
    

    def on_train_epoch_end(self):
        training_epoch_average = torch.stack(self.training_step_outputs).mean()        
        self.log("training_epoch_average", training_epoch_average, sync_dist=True)
        self.training_step_outputs.clear()  # free memory        
        # Log to TensorBoard
        self.logger.experiment.add_scalar("avg_train_loss", training_epoch_average, self.current_epoch)

    
    def on_validation_epoch_end(self):
        validation_epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", validation_epoch_average, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

        validation_epoch_average_rmse = sum(self.validation_step_outputs_rmse)/len(self.validation_step_outputs_rmse)
        self.log("validation_epoch_average_rmse", validation_epoch_average_rmse, sync_dist=True)
        self.validation_step_outputs_rmse.clear()  # free memory

        # Log to TensorBoard
        self.logger.experiment.add_scalar("avg_val_loss", validation_epoch_average, self.current_epoch)
        self.logger.experiment.add_scalar("avg_val_rmse", validation_epoch_average_rmse, self.current_epoch)


    def evaluate_inverse_scaled_rmse(self, outputs, targets, mean, std):
        outputs = outputs.detach().cpu().numpy().reshape(-1, 1)
        targets = targets.detach().cpu().numpy().reshape(-1, 1)        
        outputs_inv = outputs * std + mean
        targets_inv = targets * std + mean

        mse_inv = mean_squared_error(targets_inv, outputs_inv)
        
        return np.sqrt(mse_inv)
    
    def evaluate_rmse(self, outputs, targets):
        outputs = outputs.detach().cpu().numpy().reshape(-1, 1)
        targets = targets.detach().cpu().numpy().reshape(-1, 1)
        mse = mean_squared_error(targets, outputs)   

        return np.sqrt(mse)

############
class MCVisionNet_v2_2(pl.LightningModule):#add layer normalization after lstm
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.mean_t, self.std_t = get_train_mean_std(self.args)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.validation_step_outputs_rmse = []

        if not self.args.lstm_zero_init:
            self.SAT = True
        else:
            self.SAT = False
        
        
        self.cnn1 = models.resnet18()
        self.cnn1.fc = nn.Identity()
        self.cnn1.avgpool = nn.Identity()
        
        # self.cnn.fc = nn.Sequential(
        #     nn.Linear(self.cnn.fc.in_features, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU()
        # )

        self.cnn2 = models.resnet18()
        self.cnn2.fc = nn.Identity()
        self.cnn2.avgpool = nn.Identity()

        self.lin_proj = nn.Linear(4096*2,64)
        
        self.lstm = nn.LSTM(input_size = self.args.lstm_input_size,
                            hidden_size = self.args.lstm_hidden_units,
                            batch_first = True,
                            num_layers = self.args.lstm_num_layers,
                            dropout=0.2
        )
        self.ln = nn.LayerNorm(self.args.lstm_hidden_units)
        self.mlp = nn.Sequential(
            nn.Linear(self.args.lstm_hidden_units, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.regressor = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        self.init_h = nn.Linear(512, self.args.lstm_hidden_units)
        self.init_c = nn.Linear(512, self.args.lstm_hidden_units)

        self.criterion = torch.nn.MSELoss() 

    def initHidden(self, batch_size, enc_im=None):
        if self.args.lstm_zero_init:
            return (torch.zeros(self.args.lstm_num_layers, batch_size, self.args.lstm_hidden_units).cuda(),
                    torch.zeros(self.args.lstm_num_layers, batch_size, self.args.lstm_hidden_units).cuda())
        elif self.SAT:
            # Use the same initialization as found in Show,Attend and Tell paper
            enc_im = enc_im.mean(dim=1)
            h = self.init_h(enc_im).unsqueeze(0)
            c = self.init_c(enc_im).unsqueeze(0)

            # Ensure that h,c scales with num of stacked layers
            h = h.repeat(self.args.lstm_num_layers, 1, 1)
            c = c.repeat(self.args.lstm_num_layers, 1, 1)

            return (h, c)

    def forward(self, image1, image2, numerical):
        batch_size = numerical.shape[0]
        
        x1 = self.cnn1(image1)
        x1 = x1.reshape(batch_size, self.args.lstm_ft_map_size,512)
        x2 = self.cnn2(image2)
        x2 = x2.reshape(batch_size, self.args.lstm_ft_map_size,512)
        
        # initialize with the average of both images
        (h0_x1,c0_x1) = self.initHidden(batch_size=batch_size, enc_im=x1)
        (h0_x2,c0_x2) = self.initHidden(batch_size=batch_size, enc_im=x2)
        
        h0,c0 = (h0_x1+h0_x2),(c0_x1+c0_x2)
        # h0 = torch.zeros(num_layers, batch_size, hidden_units).requires_grad_().to(device)
        # c0 = torch.zeros(num_layers, batch_size, hidden_units).requires_grad_().to(device)
        _, (hn, _) = self.lstm(numerical, (h0, c0)) 
        # Only uses the final hidden state of the LSTM (basically not using all the other part of the sequence
        hn = self.ln(hn[0])
        x3 = hn
        # x3 = self.mlp(hn)

        x_im_combined = torch.concat([x1,x2],dim=2).view(batch_size,-1)

        x_im_combined = self.lin_proj(x_im_combined)

        # Maybe adding attention here? Which part of the images should be focused on?
        
        x = torch.cat((x_im_combined, x3), dim=1)
        x = self.regressor(x).view(-1)
        return x


    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr_initial,
            weight_decay=self.args.weight_decay
        )

        # Define the learning rate scheduler
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',  # Assuming you want to minimize a metric
                patience=self.args.lr_patience,
                factor=self.args.lr_factor,
                verbose=True  # If you want to log the LR reduction
            ),
            'name': 'lr_scheduler',  # Optional: Naming the scheduler
            'monitor': 'val_loss',  # Replace with your actual metric
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True  # Specific to ReduceLROnPlateau
        }

        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dset = MCVisionPanoDataset(args=self.args, process_type="train")

        # mean_std = train_dset.get_mean_std_for_t_scaling()
        # self.mean_t = mean_std["mean_t"]
        # self.std_t = mean_std["std_t"]

        return DataLoader(train_dset, 
                          batch_size=self.args.train_batch_size, 
                          num_workers=self.args.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          shuffle=False) 

    def val_dataloader(self):
        val_dset = MCVisionPanoDataset(args=self.args, process_type=self.args.val_type)

        return DataLoader(val_dset, 
                          batch_size=self.args.val_batch_size, 
                          num_workers=self.args.num_workers,
                          pin_memory=True, 
                          persistent_workers=True,
                          shuffle=False)
    

    def training_step(self, train_batch, batch_idx):
        images1 = train_batch["images1"]
        images2 = train_batch["images2"]
        numericals = train_batch["numericals"]
        targets = train_batch["targets"]
        outputs = self.forward(images1, images2, numericals)
        loss = self.criterion(outputs, targets)

        self.log('train_loss', loss)
        self.logger.experiment.add_scalar('train_loss', loss, self.global_step)

        self.training_step_outputs.append(loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        images1 = val_batch["images1"]
        images2 = val_batch["images2"]
        numericals = val_batch["numericals"]
        targets = val_batch["targets"]
        outputs = self.forward(images1, images2, numericals)
        loss = self.criterion(outputs, targets)

        if self.args.output_scaling:
            rmse_inv = self.evaluate_inverse_scaled_rmse(outputs, targets, self.mean_t, self.std_t)   
        else:
            rmse_inv = self.evaluate_rmse(outputs, targets)

        metrics = {
            "val_loss": loss,
            "val_RMSE": rmse_inv,
        }

        # self.log('val_loss', loss)
        # self.log('val_RMSE', rmse_inv)
        self.log_dict(metrics)

        self.logger.experiment.add_scalar('val_loss', loss, self.global_step)   
        self.logger.experiment.add_scalar('val_RMSE', rmse_inv, self.global_step)   

        self.validation_step_outputs.append(loss)
        self.validation_step_outputs_rmse.append(rmse_inv)

        return metrics
    

    def on_train_epoch_end(self):
        training_epoch_average = torch.stack(self.training_step_outputs).mean()        
        self.log("training_epoch_average", training_epoch_average, sync_dist=True)
        self.training_step_outputs.clear()  # free memory        
        # Log to TensorBoard
        self.logger.experiment.add_scalar("avg_train_loss", training_epoch_average, self.current_epoch)

    
    def on_validation_epoch_end(self):
        validation_epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", validation_epoch_average, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

        validation_epoch_average_rmse = sum(self.validation_step_outputs_rmse)/len(self.validation_step_outputs_rmse)
        self.log("validation_epoch_average_rmse", validation_epoch_average_rmse, sync_dist=True)
        self.validation_step_outputs_rmse.clear()  # free memory

        # Log to TensorBoard
        self.logger.experiment.add_scalar("avg_val_loss", validation_epoch_average, self.current_epoch)
        self.logger.experiment.add_scalar("avg_val_rmse", validation_epoch_average_rmse, self.current_epoch)


    def evaluate_inverse_scaled_rmse(self, outputs, targets, mean, std):
        outputs = outputs.detach().cpu().numpy().reshape(-1, 1)
        targets = targets.detach().cpu().numpy().reshape(-1, 1)        
        outputs_inv = outputs * std + mean
        targets_inv = targets * std + mean

        mse_inv = mean_squared_error(targets_inv, outputs_inv)
        
        return np.sqrt(mse_inv)
    
    def evaluate_rmse(self, outputs, targets):
        outputs = outputs.detach().cpu().numpy().reshape(-1, 1)
        targets = targets.detach().cpu().numpy().reshape(-1, 1)
        mse = mean_squared_error(targets, outputs)   

        return np.sqrt(mse)
    

        
