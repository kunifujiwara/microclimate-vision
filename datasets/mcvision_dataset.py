import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .datautils import get_cols, get_cols_eval, standard_scaling, robust_z_scaling, min_max_scaling, try_load_image


class MCVisionDataset(Dataset):
    def __init__(self, args, process_type):
        super().__init__()
        
        self.args = args

        suffix = ""
        for dropWS in self.args.dropwss:
            suffix = suffix + f'_{dropWS}'
        suffix_train = suffix + f"_train"
        suffix = suffix + f"_{process_type}"

        self.data = pd.read_csv(os.path.join(self.args.dataset_root, "microclimate", self.args.dataset_basename + f"{suffix}.csv"))
        data_s = pd.read_csv(os.path.join(self.args.dataset_root, "microclimate", self.args.dataset_name_s + ".csv"))

        self.pano_folder = self.args.pano_folder
        self.sate_folder = self.args.sate_folder
        self.sequence_length = self.args.lstm_sequence_length

        pcol0, pcol1, scol0, scol1, ncol0, ncol1, tcol, idxscol, ncol0_s, ncol1_s = get_cols(self.data, data_s, self.args.target_weather)

        inputs =  data_s[self.args.input_weather]

        if self.args.input_scaling:
            if args.dataset_standard == "train":
                data_standard = pd.read_csv(os.path.join(self.args.dataset_root, "microclimate", self.args.dataset_basename + f"{suffix_train}.csv"))
                input_columns = [element + "_reference" for element in args.numerical_inputs]
                inputs_standard = data_standard[input_columns]
            elif args.dataset_standard == "sequence":
                data_standard = data_s.copy()
                inputs_standard = data_standard[self.args.numerical_inputs]
            if self.args.scaling_type == "standard":
                inputs, _, _ = standard_scaling(inputs, inputs_standard)
            elif  self.args.scaling_type == "robust z":
                inputs, _, _ = robust_z_scaling(inputs, inputs_standard)
            elif  self.args.scaling_type == "min max":
                inputs, _, _ = min_max_scaling(inputs, inputs_standard)
        
        if self.args.output_scaling:
            if args.dataset_standard == "train":
                target = self.data[self.args.target_weather+"_target"]
                target_standard = data_standard[self.args.target_weather+"_reference"]
            elif args.dataset_standard == "sequence":
                target = self.data[self.args.target_weather+"_target"]
                target_standard = data_standard[self.args.target_weather]
            if self.args.scaling_type == "standard":
                target, self.stan_t_scaling, self.coef_t_scaling = standard_scaling(target, target_standard)
            elif  self.args.scaling_type == "robust z":
                target, self.stan_t_scaling, self.coef_t_scaling = robust_z_scaling(target, target_standard)
            elif  self.args.scaling_type == "min max":
                target, self.stan_t_scaling, self.coef_t_scaling = min_max_scaling(target, target_standard)

        self.pano_names1 = self.data.iloc[:, pcol0].tolist()
        self.pano_names2 = self.data.iloc[:, pcol1].tolist()        
        self.sate_names1 = self.data.iloc[:, scol0].tolist()
        self.sate_names2 = self.data.iloc[:, scol1].tolist()

        self.y = torch.tensor(target.values).float()
        self.X = torch.tensor(inputs.values).float()

        self.idx_s = self.data.iloc[:, idxscol].tolist()

        # self.times = self.data.iloc[:, timecol]
        # self.WSnames_r = self.data.iloc[:, wscol0]

        # Create a dictionary for faster lookup of idx_s
        # self.idx_s_dict = {(row.iloc[timecol-1], row.iloc[icol0-1]): idx for idx, row in data_s.iterrows()}
        # print(self.idx_s_dict)

        self.transform = transforms.Compose([
            # transforms.Resize((224,224)),
            # transforms.RandomCrop(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print("Index:", idx)
        # print("WSname_r column index (icol0-1):", icol0-1)
        # print("Dataframe shape:", self.data.shape)
        # print("Dataframe columns:", self.data.columns)
        # print("Dataframe idx_s:", self.data.columns)

        pano_name1 = os.path.join(self.pano_folder, self.pano_names1[idx])
        pano_name2 = os.path.join(self.pano_folder, self.pano_names2[idx])
        pano1 = try_load_image(pano_name1)
        pano2 = try_load_image(pano_name2)
        sate_name1 = os.path.join(self.sate_folder, self.sate_names1[idx])
        sate_name2 = os.path.join(self.sate_folder, self.sate_names2[idx])
        sate1 = try_load_image(sate_name1)
        sate2 = try_load_image(sate_name2)
        
        pano1 = self.transform(pano1)
        pano2 = self.transform(pano2)
        sate1 = self.transform(sate1)
        sate2 = self.transform(sate2)

        # time = self.times[idx]
        # WSname_r = self.WSnames_r[idx]

        # Use dictionary for faster lookup
        idx_s = self.idx_s[idx]
        #print("Dataframe idx_s:", self.data.columns)

        if idx_s >= self.sequence_length - 1:
            idx_start = idx_s - self.sequence_length + 1
            numerical = self.X[idx_start:(idx_s + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - idx_s - 1, 1)
            numerical = self.X[0:(idx_s + 1), :]
            numerical = torch.cat((padding, numerical), 0)

        target = self.y[idx]
        return {
            "panos1": pano1,
            "panos2": pano2,
            "sates1": sate1,
            "sates2": sate2,
            "numericals": numerical,
            "targets": target
        }
    
    def get_mean_std_for_t_scaling(self):
        return {
            "stan_t": self.stan_t_scaling, 
            "coef_t": self.coef_t_scaling
        }

class MCVisionDataset_sequence(Dataset):
    def __init__(self, args, process_type):
        super().__init__()
        
        self.args = args

        suffix = ""
        for dropWS in self.args.dropwss:
            suffix = suffix + f'_{dropWS}'
        suffix_train = suffix + f"_train"
        suffix = suffix + f"_{process_type}"

        #self.data = pd.read_csv(os.path.join(self.args.dataset_root, "microclimate", "sequence_"+self.args.dataset_basename + f"{suffix}.csv"))        
        self.data = pd.read_csv(os.path.join(self.args.dataset_root, "microclimate", "sequence_"+self.args.dataset_basename + f"{suffix}.csv"))
        #pd.read_csv(os.path.join(self.args.dataset_root, "microclimate", "sequence_"+self.args.dataset_basename + f".csv"))
        data_s = pd.read_csv(os.path.join(self.args.dataset_root, "microclimate", self.args.dataset_name_s + ".csv"))

        self.pano_folder = self.args.pano_folder
        self.sate_folder = self.args.sate_folder
        self.sequence_length = self.args.lstm_sequence_length

        pcol0, pcol1, scol0, scol1, ncol0, ncol1, tcol, idxscol, ncol0_s, ncol1_s = get_cols(self.data, data_s, self.args.target_weather)

        inputs =  data_s[self.args.numerical_inputs]

        if self.args.input_scaling:
            if args.dataset_standard == "train":
                data_standard = pd.read_csv(os.path.join(self.args.dataset_root, "microclimate", self.args.dataset_basename + f"{suffix_train}.csv"))
                input_columns = [element + "_reference" for element in args.numerical_inputs]
                inputs_standard = data_standard[input_columns]
            elif args.dataset_standard == "sequence":
                data_standard = data_s.copy()
                inputs_standard = data_standard[self.args.numerical_inputs]
            if self.args.scaling_type == "standard":
                inputs, _, _ = standard_scaling(inputs, inputs_standard)
            elif  self.args.scaling_type == "robust z":
                inputs, _, _ = robust_z_scaling(inputs, inputs_standard)
            elif  self.args.scaling_type == "min max":
                inputs, _, _ = min_max_scaling(inputs, inputs_standard)
        
        if self.args.output_scaling:
            if args.dataset_standard == "train":
                target = self.data[self.args.target_weather+"_target"]
                target_standard = data_standard[self.args.target_weather+"_reference"]
            elif args.dataset_standard == "sequence":
                target = self.data[self.args.target_weather+"_target"]
                target_standard = data_standard[self.args.target_weather]
            if self.args.scaling_type == "standard":
                target, self.stan_t_scaling, self.coef_t_scaling = standard_scaling(target, target_standard)
            elif  self.args.scaling_type == "robust z":
                target, self.stan_t_scaling, self.coef_t_scaling = robust_z_scaling(target, target_standard)
            elif  self.args.scaling_type == "min max":
                target, self.stan_t_scaling, self.coef_t_scaling = min_max_scaling(target, target_standard)

        self.pano_names1 = self.data.iloc[:, pcol0].tolist()
        self.pano_names2 = self.data.iloc[:, pcol1].tolist()        
        self.sate_names1 = self.data.iloc[:, scol0].tolist()
        self.sate_names2 = self.data.iloc[:, scol1].tolist()

        self.y = torch.tensor(target.values).float()
        self.X = torch.tensor(inputs.values).float()

        self.idx_s = self.data.iloc[:, idxscol].tolist()

        # self.times = self.data.iloc[:, timecol]
        # self.WSnames_r = self.data.iloc[:, wscol0]

        # Create a dictionary for faster lookup of idx_s
        # self.idx_s_dict = {(row.iloc[timecol-1], row.iloc[icol0-1]): idx for idx, row in data_s.iterrows()}
        # print(self.idx_s_dict)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print("Index:", idx)
        # print("WSname_r column index (icol0-1):", icol0-1)
        # print("Dataframe shape:", self.data.shape)
        # print("Dataframe columns:", self.data.columns)
        # print("Dataframe idx_s:", self.data.columns)

        pano_name1 = os.path.join(self.pano_folder, self.pano_names1[idx])
        pano_name2 = os.path.join(self.pano_folder, self.pano_names2[idx])
        pano1 = try_load_image(pano_name1)
        pano2 = try_load_image(pano_name2)
        sate_name1 = os.path.join(self.sate_folder, self.sate_names1[idx])
        sate_name2 = os.path.join(self.sate_folder, self.sate_names2[idx])
        sate1 = try_load_image(sate_name1)
        sate2 = try_load_image(sate_name2)
        
        pano1 = self.transform(pano1)
        pano2 = self.transform(pano2)
        sate1 = self.transform(sate1)
        sate2 = self.transform(sate2)

        # time = self.times[idx]
        # WSname_r = self.WSnames_r[idx]

        # Use dictionary for faster lookup
        idx_s = self.idx_s[idx]
        #print("Dataframe idx_s:", self.data.columns)

        if idx_s >= self.sequence_length - 1:
            idx_start = idx_s - self.sequence_length + 1
            numerical = self.X[idx_start:(idx_s + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - idx_s - 1, 1)
            numerical = self.X[0:(idx_s + 1), :]
            numerical = torch.cat((padding, numerical), 0)

        target = self.y[idx]
        return {
            "panos1": pano1,
            "panos2": pano2,
            "sates1": sate1,
            "sates2": sate2,
            "numericals": numerical,
            "targets": target
        }
    
    def get_mean_std_for_t_scaling(self):
        return {
            "stan_t": self.stan_t_scaling, 
            "coef_t": self.coef_t_scaling
        }

class MCVisionDataset_eval(Dataset):
    def __init__(self, data, args):
        super().__init__()
        
        self.args = args

        suffix = ""
        for dropWS in self.args.dropwss:
            suffix = suffix + f'_{dropWS}'
        suffix_train = suffix + f"_train"

        #self.data = pd.read_csv(os.path.join(self.args.dataset_root, "microclimate", "sequence_"+self.args.dataset_basename + f"{suffix}.csv"))
        self.data = data
        data_s = pd.read_csv(os.path.join(self.args.dataset_root, "microclimate", self.args.dataset_name_s + ".csv"))

        self.pano_folder = self.args.pano_folder
        self.pano_map_folder = self.args.pano_map_folder
        self.sate_folder = self.args.sate_folder
        self.sate_map_folder = self.args.sate_map_folder
        self.sequence_length = self.args.lstm_sequence_length

        pcol0, pcol1, scol0, scol1, ncol0, ncol1, idxscol, ncol0_s, ncol1_s = get_cols_eval(self.data, data_s, self.args.target_weather)

        inputs =  data_s[self.args.numerical_inputs]

        if self.args.input_scaling:
            if args.dataset_standard == "train":
                data_standard = pd.read_csv(os.path.join(self.args.dataset_root, "microclimate", self.args.dataset_basename + f"{suffix_train}.csv"))
                input_columns = [element + "_reference" for element in args.numerical_inputs]
                inputs_standard = data_standard[input_columns]
            elif args.dataset_standard == "sequence":
                data_standard = data_s.copy()
                inputs_standard = data_standard[self.args.numerical_inputs]
            if self.args.scaling_type == "standard":
                inputs, _, _ = standard_scaling(inputs, inputs_standard)
            elif  self.args.scaling_type == "robust z":
                inputs, _, _ = robust_z_scaling(inputs, inputs_standard)
            elif  self.args.scaling_type == "min max":
                inputs, _, _ = min_max_scaling(inputs, inputs_standard)

        self.pano_names1 = self.data.iloc[:, pcol0].tolist()
        self.pano_names2 = self.data.iloc[:, pcol1].tolist()        
        self.sate_names1 = self.data.iloc[:, scol0].tolist()
        self.sate_names2 = self.data.iloc[:, scol1].tolist()

        self.X = torch.tensor(inputs.values).float()

        self.idx_s = self.data.iloc[:, idxscol].tolist()

        # self.times = self.data.iloc[:, timecol]
        # self.WSnames_r = self.data.iloc[:, wscol0]

        # Create a dictionary for faster lookup of idx_s
        # self.idx_s_dict = {(row.iloc[timecol-1], row.iloc[icol0-1]): idx for idx, row in data_s.iterrows()}
        # print(self.idx_s_dict)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print("Index:", idx)
        # print("WSname_r column index (icol0-1):", icol0-1)
        # print("Dataframe shape:", self.data.shape)
        # print("Dataframe columns:", self.data.columns)
        # print("Dataframe idx_s:", self.data.columns)

        pano_name1 = os.path.join(self.pano_folder, self.pano_names1[idx])
        pano_name2 = os.path.join(self.pano_map_folder, self.pano_names2[idx])
        pano1 = try_load_image(pano_name1)
        pano2 = try_load_image(pano_name2)
        sate_name1 = os.path.join(self.sate_folder, self.sate_names1[idx])
        sate_name2 = os.path.join(self.sate_map_folder, self.sate_names2[idx])
        sate1 = try_load_image(sate_name1)
        sate2 = try_load_image(sate_name2)
        
        pano1 = self.transform(pano1)
        pano2 = self.transform(pano2)
        sate1 = self.transform(sate1)
        sate2 = self.transform(sate2)

        # time = self.times[idx]
        # WSname_r = self.WSnames_r[idx]

        # Use dictionary for faster lookup
        idx_s = self.idx_s[idx]
        #print("Dataframe idx_s:", self.data.columns)

        if idx_s >= self.sequence_length - 1:
            idx_start = idx_s - self.sequence_length + 1
            numerical = self.X[idx_start:(idx_s + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - idx_s - 1, 1)
            numerical = self.X[0:(idx_s + 1), :]
            numerical = torch.cat((padding, numerical), 0)

        return {
            "panos1": pano1,
            "panos2": pano2,
            "sates1": sate1,
            "sates2": sate2,
            "numericals": numerical
        }
    
    def get_mean_std_for_t_scaling(self):
        return {
            "stan_t": self.stan_t_scaling, 
            "coef_t": self.coef_t_scaling
        }