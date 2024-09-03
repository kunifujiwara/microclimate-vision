import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from models.denoise import DenoiseNet
from models.mcvision_pano import MCVisionNet_pano
from models.mcvision_sate import MCVisionNet_sate
from models.mcvision_panosate import MCVisionNet_panosate
from datasets.mcvision_dataset import MCVisionDataset
from utils.misc import get_log_dir_name_tblogger
from models.modelutils import get_train_stan_coef, get_rmse_baseline, get_optimizer, get_scheduler, train_one_epoch, validate, load_yaml_config
import argparse
import os
import sys

def main(args):
    # ... Setup logging, model initialization, etc. ...
    log_dir = args.log_dir

    # Setup TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # DataLoaders
    train_dataset = MCVisionDataset(args=args, process_type="train")
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    stan_train, coef_train = get_train_stan_coef(args)

    val_dataset = MCVisionDataset(args=args, process_type="val")
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
    val_baseline = get_rmse_baseline(args, process_type="val")

    # Model, Criterion, Optimizer
    if(args.model == "MCVisionNet_pano"):
        model = MCVisionNet_pano(args).to(args.device)
    elif(args.model == "MCVisionNet_sate"):
        model = MCVisionNet_sate(args).to(args.device)
    elif(args.model == "MCVisionNet_panosate"):
        model = MCVisionNet_panosate(args).to(args.device)

    criterion = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr_initial, weight_decay=args.weight_decay)
    optimizer = get_optimizer(args, model.parameters())
    scheduler = get_scheduler(args, optimizer)

    # Training Loop
    for epoch in range(args.train_max_epochs):
        train_loss = train_one_epoch(args, model, train_loader, optimizer, criterion, args.device)
        val_loss = validate(args, model, val_loader, criterion, args.device, stan_train, coef_train)

        # TensorBoard logging
        writer.add_scalar(f'{args.target_weather}/Loss/Train', train_loss, epoch)
        writer.add_scalar(f'{args.target_weather}/Loss/Val', val_loss["avg_loss"], epoch)
        writer.add_scalar(f'{args.target_weather}/RMSE/Val', val_loss["avg_rmse"], epoch)
        writer.add_scalar(f'{args.target_weather}/RMSE_Baseline_ratio/Val', val_loss["avg_loss"]/val_baseline, epoch)
        writer.add_scalar(f'{args.target_weather}/R2/Val', val_loss["avg_r2"], epoch)

        # Checkpointing
        if epoch % args.save_interval == 0 or epoch == args.train_max_epochs - 1:
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_epoch_{epoch}.pth'))
        
        # print(f"epoch{epoch}: Loss/Train = {train_loss:.2f}, Loss/Val = {val_loss:.2f}, RMSE/Val = {val_rmse:.2f}, RMSE_Baseline_ratio/Val = {val_rmse/val_baseline:.2f}, Loss/Test = {test_loss:.2f}, RMSE/Test = {test_rmse:.2f}, RMSE_Baseline_ratio/Test = {test_rmse/test_baseline:.2f}")
        print(f"epoch{epoch}: Loss/Train = {train_loss:.2f}, Loss/Val = {val_loss['avg_loss']:.2f}, RMSE/Val = {val_loss['avg_rmse']:.2f}, RMSE_Baseline_ratio/Val = {val_loss['avg_rmse']/val_baseline:.2f}, MAE/Val = {val_loss['avg_mae']:.2f}, R2/Val = {val_loss['avg_r2']:.2f}")

        if args.scheduler =="ReduceLROnPlateau":
            if args.lr_monitor == "train_loss":
                scheduler.step(train_loss)
            elif args.lr_monitor == "val_loss":
                scheduler.step(val_loss)
        elif args.scheduler =="MultiStepLR":
            scheduler.step()

    writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Load configuration from a YAML file")
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file', required=True)
    args, unknown = parser.parse_known_args()

    # Load configuration from YAML
    if args.config:
        config_path = args.config
        try:
            yaml_args = load_yaml_config(config_path)
        except Exception as e:
            print(f"Error reading the config file: {e}")
            sys.exit(1)            
        main(yaml_args)
    else:
        print("Please provide a path to the YAML configuration file.")
        sys.exit(1)
