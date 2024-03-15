import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import yaml
from types import SimpleNamespace


def get_train_mean_std(args):

    suffix = ""
    for dropWS in args.dropwss:
        suffix = suffix + f'_{dropWS}'
    suffix = suffix + f"_train"
    data_train = pd.read_csv(os.path.join(args.dataset_root, "microclimate", args.dataset_basename + f"{suffix}.csv"))
    tcol = data_train.columns.get_loc(args.target_weather+'_target')
    col_to_output_scale_s = data_train.columns[tcol]
    mean = data_train[col_to_output_scale_s].mean()
    std = data_train[col_to_output_scale_s].std()

    return mean, std

def get_val_rmse_baseline(args):

    suffix = ""
    for dropWS in args.dropwss:
        suffix = suffix + f'_{dropWS}'
    suffix = suffix + f"_{args.val_type}"
    data_val = pd.read_csv(os.path.join(args.dataset_root, "microclimate", args.dataset_basename + f"{suffix}.csv"))
    mse = mean_squared_error(data_val[args.target_weather+'_reference'], data_val[args.target_weather+'_target'])

    return np.sqrt(mse)


def get_optimizer(args, parameters):

    return torch.optim.Adam(
        parameters,
        lr=args.lr_initial,
        weight_decay=args.weight_decay
    )

def get_scheduler(args, optimizer):

    return {
        'scheduler': ReduceLROnPlateau(
            optimizer,
            mode='min',  # Assuming you want to minimize a metric
            patience=args.lr_patience,
            factor=args.lr_factor,
            verbose=True  # If you want to log the LR reduction
        ),
        'name': 'lr_scheduler',  # Optional: Naming the scheduler
        'monitor': 'val_loss',  # Replace with your actual metric
        'interval': 'epoch',
        'frequency': 1,
        'reduce_on_plateau': True  # Specific to ReduceLROnPlateau
    }

def evaluate_inverse_scaled_rmse(outputs, targets, mean, std):
    outputs = outputs.detach().cpu().numpy().reshape(-1, 1)
    targets = targets.detach().cpu().numpy().reshape(-1, 1)        
    outputs_inv = outputs * std + mean
    targets_inv = targets * std + mean

    mse_inv = mean_squared_error(targets_inv, outputs_inv)
    
    return np.sqrt(mse_inv)

def evaluate_rmse(outputs, targets):
    outputs = outputs.detach().cpu().numpy().reshape(-1, 1)
    targets = targets.detach().cpu().numpy().reshape(-1, 1)
    mse = mean_squared_error(targets, outputs)   

    return np.sqrt(mse)

#def get_loss_metrix(args, outputs, targets, mean_t, std_t, val_rmse_bl):
def get_log_loss_metrics(self, outputs, targets):

    loss = self.criterion(outputs, targets)
    if self.args.output_scaling:
        rmse_inv = evaluate_inverse_scaled_rmse(outputs, targets, self.mean_t, self.std_t)   
    else:
        rmse_inv = evaluate_rmse(outputs, targets)

    metrics = {
        "val_loss": loss,
        "val_RMSE": rmse_inv,
        "val_RMSE/baseline": rmse_inv/self.val_rmse_bl,
    }

    self.log_dict(metrics)

    self.logger.experiment.add_scalar('val_loss', loss, self.global_step)   
    self.logger.experiment.add_scalar('val_RMSE', rmse_inv, self.global_step)
    self.logger.experiment.add_scalar('val_RMSE/baseline', rmse_inv/self.val_rmse_bl, self.global_step)      

    self.validation_step_outputs.append(loss)
    self.validation_step_outputs_rmse.append(rmse_inv)

    return metrics

def log_loss_average(self):
    validation_epoch_average = torch.stack(self.validation_step_outputs).mean()
    self.log("validation_epoch_average", validation_epoch_average, sync_dist=True)
    self.validation_step_outputs.clear()  # free memory

    validation_epoch_average_rmse = sum(self.validation_step_outputs_rmse)/len(self.validation_step_outputs_rmse)
    self.log("validation_epoch_average_rmse", validation_epoch_average_rmse, sync_dist=True)
    self.validation_step_outputs_rmse.clear()  # free memory

    # Log to TensorBoard
    self.logger.experiment.add_scalar("avg_val_loss", validation_epoch_average, self.current_epoch)
    self.logger.experiment.add_scalar("avg_val_rmse", validation_epoch_average_rmse, self.current_epoch)
    self.logger.experiment.add_scalar("avg_val_rmse/baseline", validation_epoch_average_rmse/self.val_rmse_bl, self.current_epoch)

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        args = yaml.safe_load(file)
        return SimpleNamespace(**args)
    
def train_one_epoch(args, model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    if args.tag == "oneref":
        for train_batch in train_loader:
            panos = train_batch["panos"].to(device)
            sates = train_batch["sates"].to(device)
            numericals = train_batch["numericals"].to(device)
            targets = train_batch["targets"].to(device)
            
            optimizer.zero_grad()
            if "panosate" in args.model:
                outputs = model(panos, sates, numericals)
            elif "pano" in args.model:
                outputs = model(panos, numericals)
            elif "sate" in args.model:
                outputs = model(sates, numericals)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()   
    elif args.tag == "oneref_delta_lstm":
        for train_batch in train_loader:
            panos = train_batch["panos"].to(device)
            sates = train_batch["sates"].to(device)
            numericals = train_batch["numericals"].to(device)
            targets = train_batch["targets"].to(device)
            
            optimizer.zero_grad()
            if "panosate" in args.model:
                outputs = model(panos, sates, numericals)
            elif "pano" in args.model:
                outputs = model(panos, numericals)
            elif "sate" in args.model:
                outputs = model(sates, numericals)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
    elif args.tag == "oneref_delta":
        for train_batch in train_loader:
            panos = train_batch["panos"].to(device)
            sates = train_batch["sates"].to(device)
            targets = train_batch["targets"].to(device)
            
            optimizer.zero_grad()
            if "panosate" in args.model:
                outputs = model(panos, sates)
            elif "pano" in args.model:
                outputs = model(panos)
            elif "sate" in args.model:
                outputs = model(sates)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()    
    elif args.tag == "oneref_delta_hour":
        for train_batch in train_loader:
            panos = train_batch["panos"].to(device)
            sates = train_batch["sates"].to(device)
            hours = train_batch["hours"].to(device)
            targets = train_batch["targets"].to(device)
            
            optimizer.zero_grad()
            if "panosate" in args.model:
                outputs = model(panos, sates, hours)
            elif "pano" in args.model:
                outputs = model(panos, hours)
            elif "sate" in args.model:
                outputs = model(sates, hours)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()     
    elif args.tag == "oneref_delta_hour":
        for train_batch in train_loader:
            panos = train_batch["panos"].to(device)
            sates = train_batch["sates"].to(device)
            hours = train_batch["hours"].to(device)
            targets = train_batch["targets"].to(device)
            
            optimizer.zero_grad()
            if "panosate" in args.model:
                outputs = model(panos, sates, hours)
            elif "pano" in args.model:
                outputs = model(panos, hours)
            elif "sate" in args.model:
                outputs = model(sates, hours)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  
    else:
        for train_batch in train_loader:
            panos1 = train_batch["panos1"].to(device)
            panos2 = train_batch["panos2"].to(device)
            sates1 = train_batch["sates1"].to(device)
            sates2 = train_batch["sates2"].to(device)
            numericals = train_batch["numericals"].to(device)
            targets = train_batch["targets"].to(device)
            
            optimizer.zero_grad()
            if "panosate" in args.model:
                outputs = model(panos1, panos2, sates1, sates2, numericals)
            elif "pano" in args.model:
                outputs = model(panos1, panos2, numericals)
            elif "sate" in args.model:
                outputs = model(sates1, sates2, numericals)
            elif "lstm" in args.model:
                outputs = model(numericals)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()   
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(args, model, val_loader, criterion, device, standard, coefficient):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_rmse = 0
    total_mae = 0
    total_r2 = 0
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        if args.tag == "oneref":
            for val_batch in val_loader:
                panos = val_batch["panos"].to(device)
                sates = val_batch["sates"].to(device)
                numericals = val_batch["numericals"].to(device)
                targets = val_batch["targets"].to(device)
                if "panosate" in args.model:
                    outputs = model(panos, sates, numericals)
                elif "pano" in args.model:
                    outputs = model(panos, numericals)
                elif "sate" in args.model:
                    outputs = model(sates, numericals)
                loss = criterion(outputs, targets)
                #rmse = get_rmse(args, outputs, targets, standard, coefficient)
                error = get_error_metrix(args, outputs, targets, standard, coefficient)

                # Update running validation loss
                total_loss += loss.item()
                total_rmse += error["rmse"]
                total_mae += error["mae"]  
                total_r2 += error["R2"]     
        elif args.tag == "oneref_delta_lstm":
            for val_batch in val_loader:
                panos = val_batch["panos"].to(device)
                sates = val_batch["sates"].to(device)
                numericals = val_batch["numericals"].to(device)
                targets = val_batch["targets"].to(device)
                if "panosate" in args.model:
                    outputs = model(panos, sates, numericals)
                elif "pano" in args.model:
                    outputs = model(panos, numericals)
                elif "sate" in args.model:
                    outputs = model(sates, numericals)
                loss = criterion(outputs, targets)
                #rmse = get_rmse(args, outputs, targets, standard, coefficient)
                error = get_error_metrix(args, outputs, targets, standard, coefficient)

                # Update running validation loss
                total_loss += loss.item()
                total_rmse += error["rmse"]
                total_mae += error["mae"]  
                total_r2 += error["R2"]     
        elif args.tag == "oneref_delta":
            for val_batch in val_loader:
                panos = val_batch["panos"].to(device)
                sates = val_batch["sates"].to(device)
                targets = val_batch["targets"].to(device)
                if "panosate" in args.model:
                    outputs = model(panos, sates)
                elif "pano" in args.model:
                    outputs = model(panos)
                elif "sate" in args.model:
                    outputs = model(sates)
                loss = criterion(outputs, targets)
                #rmse = get_rmse(args, outputs, targets, standard, coefficient)
                error = get_error_metrix(args, outputs, targets, standard, coefficient)

                # Update running validation loss
                total_loss += loss.item()
                total_rmse += error["rmse"]
                total_mae += error["mae"]  
                total_r2 += error["R2"]     
        elif args.tag == "oneref_delta_hour":
            for val_batch in val_loader:
                panos = val_batch["panos"].to(device)
                sates = val_batch["sates"].to(device)
                hours = val_batch["hours"].to(device)
                targets = val_batch["targets"].to(device)
                if "panosate" in args.model:
                    outputs = model(panos, sates, hours)
                elif "pano" in args.model:
                    outputs = model(panos, hours)
                elif "sate" in args.model:
                    outputs = model(sates, hours)
                loss = criterion(outputs, targets)
                #rmse = get_rmse(args, outputs, targets, standard, coefficient)
                error = get_error_metrix(args, outputs, targets, standard, coefficient)

                # Update running validation loss
                total_loss += loss.item()
                total_rmse += error["rmse"]
                total_mae += error["mae"]  
                total_r2 += error["R2"]     
        else:
            for val_batch in val_loader:
                panos1 = val_batch["panos1"].to(device)
                panos2 = val_batch["panos2"].to(device)
                sates1 = val_batch["sates1"].to(device)
                sates2 = val_batch["sates2"].to(device)
                numericals = val_batch["numericals"].to(device)
                targets = val_batch["targets"].to(device)
                if "panosate" in args.model:
                    outputs = model(panos1, panos2, sates1, sates2, numericals)
                elif "pano" in args.model:
                    outputs = model(panos1, panos2, numericals)
                elif "sate" in args.model:
                    outputs = model(sates1, sates2, numericals)
                elif "lstm" in args.model:
                    outputs = model(numericals)
                loss = criterion(outputs, targets)
                #rmse = get_rmse(args, outputs, targets, standard, coefficient)
                error = get_error_metrix(args, outputs, targets, standard, coefficient)

                # Update running validation loss
                total_loss += loss.item()
                total_rmse += error["rmse"]
                total_mae += error["mae"]  
                total_r2 += error["R2"]     

    # Calculate average loss over an epoch
    loss_metrix = {
        "avg_loss" : total_loss / len(val_loader),
        "avg_rmse" : total_rmse / len(val_loader),
        "avg_mae" : total_mae / len(val_loader),
        "avg_r2" : total_r2 / len(val_loader),
    }

    return loss_metrix
    

def test(args, model, test_loader, criterion, stan, coef):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_rmse = 0
    output_list = []
    with torch.no_grad():  # Turn off gradients for testing
        if args.tag == "oneref":
            for test_batch in test_loader:
                panos = test_batch["panos"].to(args.device)
                sates = test_batch["sates"].to(args.device)
                numericals = test_batch["numericals"].to(args.device)
                targets = test_batch["targets"].to(args.device)
                if "panosate" in args.model:
                    outputs = model(panos, sates, numericals)
                elif "pano" in args.model:
                    outputs = model(panos, numericals)
                elif "sate" in args.model:
                    outputs = model(sates, numericals)
                loss = criterion(outputs, targets)
                rmse = get_rmse(args, outputs, targets, stan, coef)

                outputs_inv = get_inverse_scaled_value(outputs, stan, coef)
                output_list.extend(outputs_inv)

                # Update running validation loss
                total_loss += loss.item()
                total_rmse += rmse 
        else:
            for test_batch in test_loader:
                panos1 = test_batch["panos1"].to(args.device)
                panos2 = test_batch["panos2"].to(args.device)
                sates1 = test_batch["sates1"].to(args.device)
                sates2 = test_batch["sates2"].to(args.device)
                numericals = test_batch["numericals"].to(args.device)
                targets = test_batch["targets"].to(args.device)                
                if "panosate" in args.model:
                    outputs = model(panos1, panos2, sates1, sates2, numericals)
                elif "pano" in args.model:
                    outputs = model(panos1, panos2, numericals)
                elif "sate" in args.model:
                    outputs = model(sates1, sates2, numericals)
                loss = criterion(outputs, targets)
                rmse = get_rmse(args, outputs, targets, stan, coef)

                outputs_inv = get_inverse_scaled_value(outputs, stan, coef)
                output_list.extend(outputs_inv)

                total_loss += loss.item()
                total_rmse += rmse

    avg_loss = total_loss / len(test_loader)
    avg_rmse = total_rmse / len(test_loader)
    return avg_loss, avg_rmse, output_list

def evaluate(args, model, eval_loader, stan, coef):
    model.eval()  # Set the model to evaluation mode
    output_list = []
    with torch.no_grad():  # Turn off gradients for testing
        for eval_batch in eval_loader:
            panos1 = eval_batch["panos1"].to(args.device)
            panos2 = eval_batch["panos2"].to(args.device)
            sates1 = eval_batch["sates1"].to(args.device)
            sates2 = eval_batch["sates2"].to(args.device)
            numericals = eval_batch["numericals"].to(args.device)
                           
            if "panosate" in args.model:
                outputs = model(panos1, panos2, sates1, sates2, numericals)
            elif "pano" in args.model:
                outputs = model(panos1, panos2, numericals)
            elif "sate" in args.model:
                outputs = model(sates1, sates2, numericals)

            outputs_inv = get_inverse_scaled_value(outputs, stan, coef)
            output_list.extend(outputs_inv)
    return output_list

def evaluate_inverse_scaled_rmse(outputs, targets, standard, coefficient):
    outputs_np = outputs.detach().cpu().numpy().reshape(-1, 1)
    targets_np = targets.detach().cpu().numpy().reshape(-1, 1)        
    outputs_inv = outputs_np * coefficient + standard
    targets_inv = targets_np * coefficient + standard

    mse_inv = mean_squared_error(targets_inv, outputs_inv)
    
    return np.sqrt(mse_inv)

def evaluate_inverse_scaled_error_metrix(outputs, targets, standard, coefficient):
    outputs_np = outputs.detach().cpu().numpy().reshape(-1, 1)
    targets_np = targets.detach().cpu().numpy().reshape(-1, 1)        
    outputs_inv = outputs_np * coefficient + standard
    targets_inv = targets_np * coefficient + standard

    mse_inv = mean_squared_error(targets_inv, outputs_inv)
    rmse_inv = np.sqrt(mse_inv)
    mae_inv = mean_absolute_error(targets_inv, outputs_inv)
    r2_inv = r2_score(targets_inv, outputs_inv)

    return {"rmse":rmse_inv, "mae": mae_inv, "R2": r2_inv}

def get_inverse_scaled_value(values, standard, coefficient):
    values_np = values.detach().cpu().numpy()  
    values_inv = values_np * coefficient + standard    
    return values_inv

def evaluate_rmse(outputs, targets):
    outputs = outputs.detach().cpu().numpy().reshape(-1, 1)
    targets = targets.detach().cpu().numpy().reshape(-1, 1)
    mse = mean_squared_error(targets, outputs)   

    return np.sqrt(mse)

def evaluate_error_metrix(outputs, targets):
    outputs = outputs.detach().cpu().numpy().reshape(-1, 1)
    targets = targets.detach().cpu().numpy().reshape(-1, 1)
    mse = mean_squared_error(targets, outputs)   
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, outputs)
    r2 = r2_score(targets, outputs)

    return {"rmse":rmse, "mae": mae, "R2": r2}

#def get_loss_metrix(args, outputs, targets, mean_t, std_t, val_rmse_bl):
def get_rmse(args, outputs, targets, standard, coefficient):

    if args.output_scaling:
        rmse = evaluate_inverse_scaled_rmse(outputs, targets, standard, coefficient)   
    else:
        rmse = evaluate_rmse(outputs, targets)

    return rmse

def get_error_metrix(args, outputs, targets, standard, coefficient):

    if args.output_scaling:
        error = evaluate_inverse_scaled_error_metrix(outputs, targets, standard, coefficient)   
    else:
        error = evaluate_error_metrix(outputs, targets)

    return error

def get_train_stan_coef(args):

    suffix = ""
    for dropWS in args.dropwss:
        suffix = suffix + f'_{dropWS}'
    suffix = suffix + f"_train"
    data_train = pd.read_csv(os.path.join(args.dataset_root, "microclimate", args.dataset_basename + f"{suffix}.csv"))
    tcol = data_train.columns.get_loc(args.target_weather+'_target')
    col_to_output_scale_s = data_train.columns[tcol]
    if args.scaling_type == "standard":
        stan = data_train[col_to_output_scale_s].mean()
        coef = data_train[col_to_output_scale_s].std()
    elif args.scaling_type == "robust z":
        stan = data_train[col_to_output_scale_s].median()
        coef = data_train[col_to_output_scale_s].quantile(0.75) - data_train[col_to_output_scale_s].quantile(0.25)

    return stan, coef

def get_rmse_baseline(args, process_type):

    suffix = ""
    for dropWS in args.dropwss:
        suffix = suffix + f'_{dropWS}'
    suffix = suffix + f"_{process_type}"
    data_val = pd.read_csv(os.path.join(args.dataset_root, "microclimate", args.dataset_basename + f"{suffix}.csv"))
    mse = mean_squared_error(data_val[args.target_weather+'_reference'], data_val[args.target_weather+'_target'])

    return np.sqrt(mse)

def get_optimizer(args, parameters):

    if args.optimizer == "Adam":
        return torch.optim.Adam(
            parameters,
            lr=args.lr_initial,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "SGD":
        return torch.optim.SGD(
            parameters, 
            lr=args.lr_initial, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay)

def get_scheduler(args, optimizer):

    if args.scheduler == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
                optimizer,
                mode='min',  # Assuming you want to minimize a metric
                patience=args.lr_patience,
                factor=args.lr_factor,
                verbose=True  # If you want to log the LR reduction
            )
    elif args.scheduler == "MultiStepLR":
        return MultiStepLR(optimizer, [30, 70, 90], args.lr_factor)

def get_loss_function(args):

    if args.loss_function == "MSE":
        criterion = nn.MSELoss()
    elif args.loss_function == "MAE":
        criterion = nn.L1Loss()
    elif args.loss_function == "Huber":
        criterion = nn.SmoothL1Loss()
    return criterion

