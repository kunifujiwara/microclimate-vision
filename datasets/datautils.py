import random
import numpy as np
import torch
import torch.nn as nn
import time
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os

# class Datautils():

def get_cols_pano(data, data_s, weather_item):

    wscol0 = data.columns.get_loc('WSname_reference')
    icol0 = data.columns.get_loc('IMGname_reference')
    icol1 = data.columns.get_loc('IMGname_target')
    ncol0 = data.columns.get_loc("solar radiation_reference")
    ncol1 = data.columns.get_loc("wind speed_reference")
    tcol = data.columns.get_loc(weather_item+'_target')
    idxscol = data.columns.get_loc("idx_s")
    timecol = data.columns.get_loc("time_reference")
    ncol0_s = data_s.columns.get_loc("solar radiation")
    #ncol1_s = data_s.columns.get_loc("wind speed")
    ncol1_s = data_s.columns.get_loc("wind direction")
    tcol_s = data_s.columns.get_loc(weather_item)
    
    return wscol0, icol0, icol1, ncol0, ncol1, tcol, idxscol, timecol, ncol0_s, ncol1_s, tcol_s

def get_cols(data, data_s, weather_item):

    #wscol0 = data.columns.get_loc('WSname_reference')
    pcol0 = data.columns.get_loc('panoIMGname_reference')
    pcol1 = data.columns.get_loc('panoIMGname_target')
    scol0 = data.columns.get_loc('sateIMGname_reference')
    scol1 = data.columns.get_loc('sateIMGname_target')
    ncol0 = data.columns.get_loc("solar radiation_reference")
    #ncol1 = data.columns.get_loc("wind speed_reference")
    ncol1 = data.columns.get_loc("wind direction_reference")
    tcol = data.columns.get_loc(weather_item+'_target')
    #timecol = data.columns.get_loc("time_reference")
    idxscol = data.columns.get_loc("idx_s")
    ncol0_s = data_s.columns.get_loc("solar radiation")
    #ncol1_s = data_s.columns.get_loc("wind speed")
    ncol1_s = data_s.columns.get_loc("wind direction")
    #tcol_s = data_s.columns.get_loc(weather_item)
    
    return pcol0, pcol1, scol0, scol1, ncol0, ncol1, tcol, idxscol, ncol0_s, ncol1_s

def get_cols_eval(data, data_s, weather_item):

    #wscol0 = data.columns.get_loc('WSname_reference')
    pcol0 = data.columns.get_loc('panoIMGname_reference')
    pcol1 = data.columns.get_loc('panoIMGname_target')
    scol0 = data.columns.get_loc('sateIMGname_reference')
    scol1 = data.columns.get_loc('sateIMGname_target')
    ncol0 = data.columns.get_loc("solar radiation_reference")
    #ncol1 = data.columns.get_loc("wind speed_reference")
    ncol1 = data.columns.get_loc("wind direction_reference")
    #timecol = data.columns.get_loc("time_reference")
    idxscol = data.columns.get_loc("idx_s")
    ncol0_s = data_s.columns.get_loc("solar radiation")
    #ncol1_s = data_s.columns.get_loc("wind speed")
    ncol1_s = data_s.columns.get_loc("wind direction")
    #tcol_s = data_s.columns.get_loc(weather_item)
    
    return pcol0, pcol1, scol0, scol1, ncol0, ncol1, idxscol, ncol0_s, ncol1_s

def get_cols_oneref(data, data_s, weather_item):

    #wscol0 = data.columns.get_loc('WSname_reference')
    pcol = data.columns.get_loc('panoIMGname_target')
    scol = data.columns.get_loc('sateIMGname_target')
    ncol0 = data.columns.get_loc("solar radiation_reference")
    #ncol1 = data.columns.get_loc("wind speed_reference")
    ncol1 = data.columns.get_loc("wind direction_reference")
    tcol = data.columns.get_loc(weather_item+'_target')
    #timecol = data.columns.get_loc("time_reference")
    idxscol = data.columns.get_loc("idx_s")
    ncol0_s = data_s.columns.get_loc("solar radiation")
    #ncol1_s = data_s.columns.get_loc("wind speed")
    ncol1_s = data_s.columns.get_loc("wind direction")
    #tcol_s = data_s.columns.get_loc(weather_item)
    
    return pcol, scol, ncol0, ncol1, tcol, idxscol, ncol0_s, ncol1_s

def get_cols_oneref_v2(data, weather_item):

    #wscol0 = data.columns.get_loc('WSname_reference')
    pcol = data.columns.get_loc('panoIMGname_target')
    scol = data.columns.get_loc('sateIMGname_target')
    tcol0 = data.columns.get_loc(weather_item+'_reference')
    tcol1 = data.columns.get_loc(weather_item+'_target')
    #timecol = data.columns.get_loc("time_reference")
    #tcol_s = data_s.columns.get_loc(weather_item)
    
    return pcol, scol, tcol0, tcol1

def input_standard_scaling(df_target, start_t, end_t, df_standard, start_s, end_s):

    #calculate mean and std of df_standard
    cols_to_input_scale_idx_s = range(start_s, end_s)
    cols_to_input_scale_s = df_standard.columns[cols_to_input_scale_idx_s]
    means = df_standard[cols_to_input_scale_s].mean().values
    stds = df_standard[cols_to_input_scale_s].std().values

    #scaling
    cols_to_input_scale_idx_t = range(start_t, end_t)
    cols_to_input_scale_t = df_target.columns[cols_to_input_scale_idx_t]       
    df_target[cols_to_input_scale_t] = (df_target[cols_to_input_scale_t] - means) / stds

def output_standard_scaling(df_target, tcol_t, df_standard, tcol_s):

    col_to_output_scale_s = df_standard.columns[tcol_s]
    mean = df_standard[col_to_output_scale_s].mean()
    std = df_standard[col_to_output_scale_s].std()

    col_to_input_scale_t = df_standard.columns[tcol_t]
    df_target[col_to_input_scale_t] = (df_target[col_to_input_scale_t] - mean) / std
    return mean, std

def input_robust_z_scaling(df_target, start_t, end_t, df_standard, start_s, end_s):

    #calculate mean and std of df_standard
    cols_to_input_scale_idx_s = range(start_s, end_s)
    cols_to_input_scale_s = df_standard.columns[cols_to_input_scale_idx_s]
    medians = df_standard[cols_to_input_scale_s].median().values
    iqrs = df_standard[cols_to_input_scale_s].quantile(0.75).values - df_standard[cols_to_input_scale_s].quantile(0.25).values

    #scaling
    cols_to_input_scale_idx_t = range(start_t, end_t)
    cols_to_input_scale_t = df_target.columns[cols_to_input_scale_idx_t]       
    df_target[cols_to_input_scale_t] = (df_target[cols_to_input_scale_t] - medians) / iqrs

def standard_scaling(df_target, df_standard):
    #calculate mean and std of df_standard
    if type(df_target) == pd.core.frame.DataFrame:
        means = df_standard.mean().values
        stds = df_standard.std().values
    else:
        means = df_standard.mean()
        stds = df_standard.std()

    #scaling      
    df_target = (df_target - means) / stds
    return df_target, means, stds

def robust_z_scaling(df_target, df_standard):
    #calculate mean and std of df_standard
    if type(df_target) == pd.core.frame.DataFrame:
        medians = df_standard.median().values
        iqrs = df_standard.quantile(0.75).values - df_standard.quantile(0.25).values
    else:
        medians = df_standard.median()
        iqrs = df_standard.quantile(0.75) - df_standard.quantile(0.25)

    #scaling       
    df_target = (df_target - medians) / iqrs
    return df_target, medians, iqrs

def min_max_scaling(df_target, df_reference):
    # Calculate min and max of df_reference
    if isinstance(df_target, pd.DataFrame):
        mins = df_reference.min().values
        maxs = df_reference.max().values
    else:
        mins = df_reference.min()
        maxs = df_reference.max()

    # Scaling
    df_target = (df_target - mins) / (maxs - mins)
    return df_target, mins, (maxs - mins)

def output_robust_z_scaling(df_target, tcol_t, df_standard, tcol_s):
    col_to_output_scale_s = df_standard.columns[tcol_s]
    median = df_standard[col_to_output_scale_s].median()
    iqr = df_standard[col_to_output_scale_s].quantile(0.75) - df_standard[col_to_output_scale_s].quantile(0.25)
    col_to_input_scale_t = df_standard.columns[tcol_t]
    df_target[col_to_input_scale_t] = (df_target[col_to_input_scale_t] - median) / iqr

    return median, iqr

def read_data(args, process_type):

    if process_type == "train":
        data = pd.read_csv(args.training_data_path)
    elif process_type == "val":
        data = pd.read_csv(args.validation_data_path)
    elif process_type == "test":            
        data = pd.read_csv(args.test_data_path)

    return data

def try_load_image(image_path, max_attempts=3, sleep_duration=0.1):
    attempts = 0
    while attempts < max_attempts:
        try:
            image = Image.open(image_path)
            return image
        except UnidentifiedImageError:
            attempts += 1
            time.sleep(sleep_duration)
            if attempts >= max_attempts:
                print(f"Failed to load image after {max_attempts} attempts: {image_path}")
                return None  # Or handle as required
    return None

