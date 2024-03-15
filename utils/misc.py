import os
import torch
import numpy as np
import pandas as pd
import random
import time
import logging
import logging.handlers
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import yaml

THOUSAND = 1000
MILLION = 1000000


class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def int_list(argstr):
    return list(map(int, argstr.split(',')))


def str_list(argstr):
    return list(argstr.split(','))


def get_log_dir_name_tblogger(name=''):
    log_dir_name = name + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    return log_dir_name


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def represent_str(dumper, data):
    """ Custom representation of strings for YAML. """
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

# def log_hyperparams(writer, log_dir, args):
#     from torch.utils.tensorboard.summary import hparams

#     # Convert all values to strings, handling non-string types appropriately
#     vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}

#     # Add custom string representation
#     yaml.add_representer(str, represent_str)

#     # Logging hyperparameters to TensorBoard
#     exp, ssi, sei = hparams(vars_args, {"hp_metric": -1})
#     fw = writer._get_file_writer()
#     fw.add_summary(exp)
#     fw.add_summary(ssi)
#     fw.add_summary(sei)

#     # Writing hyperparameters to a YAML file
#     with open(os.path.join(log_dir, 'hparams.yaml'), 'w') as yamlf:
#         yaml.dump(vars_args, yamlf, default_flow_style=False)

def custom_yaml_representer(dumper, data):
    if isinstance(data, (bool, int, float)):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))
    elif isinstance(data, list):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

def log_hyperparams(log_dir, args):
    config = vars(args)
    # Add custom representer to the YAML dumper
    yaml.add_representer(str, custom_yaml_representer)
    yaml.add_representer(list, custom_yaml_representer)
    # Write the configuration to a YAML file
    with open(os.path.join(log_dir, 'hparams.yaml'), 'w') as yamlf:
        yaml.dump(config, yamlf)

# def log_hyperparams(writer, log_dir, args):
#     from torch.utils.tensorboard.summary import hparams

#     # Convert all values to strings, handling non-string types appropriately
#     vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}

#     # Logging hyperparameters to TensorBoard
#     exp, ssi, sei = hparams(vars_args, {"hp_metric": -1})
#     fw = writer._get_file_writer()
#     fw.add_summary(exp)
#     fw.add_summary(ssi)
#     fw.add_summary(sei)

#     # Writing hyperparameters to a YAML file
#     with open(os.path.join(log_dir, 'hparams.yaml'), 'w') as yamlf:
#         yaml.dump(vars_args, yamlf, default_flow_style=False)

# def log_hyperparams(writer, log_dir, args):
#     from torch.utils.tensorboard.summary import hparams
#     vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
#     exp, ssi, sei = hparams(vars_args, {"hp_metric": -1})
#     fw = writer._get_file_writer()
#     fw.add_summary(exp)
#     fw.add_summary(ssi)
#     fw.add_summary(sei)
#     with open(os.path.join(log_dir, 'hparams.csv'), 'w') as csvf:
#         csvf.write('key,value\n')
#         for k, v in vars_args.items():
#             csvf.write('%s,%s\n' % (k, v))

# def log_hyperparams(logger, args):
#     vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
#     logger.log_hyperparams(vars_args)
    

def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def parse_experiment_name(name):
    if 'blensor' in name:
        if 'Ours' in name:
            dataset, method, tag, blensor_, noise = name.split('_')[:5]
        else:
            dataset, method, blensor_, noise = name.split('_')[:4]
        return {
            'dataset': dataset,
            'method': method,
            'resolution': 'blensor',
            'noise': noise,
        }

    if 'real' in name:
        if 'Ours' in name:
            dataset, method, tag, blensor_, noise = name.split('_')[:5]
        else:
            dataset, method, blensor_, noise = name.split('_')[:4]
        return {
            'dataset': dataset,
            'method': method,
            'resolution': 'real',
            'noise': noise,
        }
        
    else:
        if 'Ours' in name:
            dataset, method, tag, num_pnts, sample_method, noise = name.split('_')[:6]
        else:
            dataset, method, num_pnts, sample_method, noise = name.split('_')[:5]
        return {
            'dataset': dataset,
            'method': method,
            'resolution': num_pnts + '_' + sample_method,
            'noise': noise,
        }

def get_error_baseline(data, target_weather):

    mse = mean_squared_error(data[target_weather+'_reference'], data[target_weather+'_target'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(data[target_weather+'_reference'], data[target_weather+'_target'])
    r2 = r2_score(data[target_weather+'_reference'], data[target_weather+'_target'])

    return {"rmse":rmse, "mae": mae, "R2": r2}

