import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from models.denoise import DenoiseNet
from models.mcvision_pano import MCVisionNet_pano
from models.mcvision_sate import MCVisionNet_sate
from models.mcvision_panosate import MCVisionNet_panosate
#from datasets.mcvision_dataset import MCVisionDataset, MCVisionDataset_sequence
from datasets.mcvision_dataset import MCVisionDataset
from models.modelutils import get_train_stan_coef, get_rmse_baseline
import argparse
import sys
import os
from models.modelutils import load_yaml_config, test
from datasets.datautils import read_data



def main(args_command):

    args = load_yaml_config(args_command.config)

    # ... Setup logging, model initialization, etc. ...
    stan_train, coef_train = get_train_stan_coef(args)

    print(f"Loading test dataset")
    process_type = "test"
    test_dataset = MCVisionDataset(args=args, process_type=process_type)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_baseline = get_rmse_baseline(args, process_type=process_type)

    # Model and Criterion
    # Model, Criterion, Optimizer
    print(f"Loading model")
    if(args.model == "MCVisionNet_pano"):
        model = MCVisionNet_pano(args).to(args.device)
    elif(args.model == "MCVisionNet_sate"):
        model = MCVisionNet_sate(args).to(args.device)
    elif(args.model == "MCVisionNet_panosate"):
        model = MCVisionNet_panosate(args).to(args.device)
    criterion = nn.MSELoss()

    # Load your model weights
    state_dict = torch.load(args_command.model)
    model.load_state_dict(state_dict)

    ## Testing
    print(f"Conducting inference and calculating loss metrix")
    test_loss, test_rmse, output_list = test(args, model, test_loader, criterion, stan_train, coef_train)
    print(f"Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test RMSE/baseline: {test_rmse/test_baseline:.4f}")
    data_test = read_data(args, process_type = process_type)
    data_test[f"{args.target_weather}_target_pred"]=output_list
    data_test.to_csv(args_command.result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load data and model")
    parser.add_argument('--config', type=str, help='Path to config file', required=True)
    parser.add_argument('--model', type=str, help='Path to model weight', required=True)
    parser.add_argument('--result', type=str, help='Path to prediction result', required=True)
    args0, unknown = parser.parse_known_args()

    # Load configuration from YAML
    if args0.config:
        if args0.model:
            if args0.result:
                main(args0)
            else:
                print("Please provide a path to save prediction result")
                sys.exit(1)
        else:
            print("Please provide a path to the model weight")
            sys.exit(1)
    else:
        print("Please provide a path to the config file")
        sys.exit(1)
