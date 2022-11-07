import json
import os
import random
import shutil
import time
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import configs.config as config




def reproducibility():
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    cudnn.deterministic = config.deterministic
    cudnn.benchmark = config.benchmark



def save_arguments(args, path):
    with open(path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()



def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)



def datestr():
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))



def save_list(name, list):
    with open(name, "wb") as fp:
        pickle.dump(list, fp)



def load_list(name):
    with open(name, "rb") as fp:
        list_file = pickle.load(fp)

    return list_file




def prepare_input(input_tuple, device=None):
    input_tensor, target = input_tuple

    if config.cuda:
        input_tensor, target = input_tensor.to(device), target.to(device)

    return input_tensor, target



def load_json_file(file_path):
    def key_2_int(x):
        return {int(k): v for k, v in x.items()}

    json_file = r"./lib/dataloaders/index_to_class/3DTooth.json"
    assert os.path.exists(file_path), "{} file not exist.".format(file_path)
    json_file = open(file_path, 'r')
    dict = json.load(json_file, object_hook=key_2_int)
    json_file.close()

    return dict




