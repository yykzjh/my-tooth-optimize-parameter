import os
import torch.nn as nn
import torch.optim as optim

import configs.config as config
from lib.models.DenseVNet import DenseVNet


model_list = ['DENSEVNET', 'VNET']

optimizer_list = ['sgd', 'adam', 'rmsprop']



def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



def create_model(args):

    # 创建模型
    if config.model_name == 'DENSEVNET':
        model = DenseVNet(in_channels=config.in_channels, classes=config.classes)

    else:
        raise RuntimeError(f"Unsupported model: '{config.model_name}'. Supported models: {model_list}")

    # 打印模型大小，即参数量
    print(config.model_name, 'Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))



    # 创建优化器
    if config.optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

    elif config.optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    elif config.optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    else:
        raise RuntimeError(f"Unsupported optimizer: '{config.optimizer_name}'. Supported optimizers: {optimizer_list}")


    # 随机初始化模型参数
    model.apply(weight_init)


    # 加载预训练权重
    if args.pretrain is not None:
        if os.path.isfile(args.pretrain):
            print("=> 加载预训练权重 '{}'".format(args.pretrain))
            checkpoint = model.load_checkpoint(args.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise RuntimeError(f"no pretrained model found at '{args.pretrain}'")


    # 加载检查点参数及模型权重
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> 加载检查点 '{}'".format(args.resume))
            checkpoint = model.load_checkpoint(args.resume)
            args.sEpoch = checkpoint['epoch']
            args.best_dice = checkpoint['best_metric']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            raise RuntimeError(f"no checkpoint found at '{args.resume}'")


    return model, optimizer





def create_test_model(pretrain=None):

    # 创建模型
    if config.model_name == 'DENSEVNET':
        model = DenseVNet(in_channels=config.in_channels, classes=config.classes)

    else:
        raise RuntimeError(f"Unsupported model: '{config.model_name}'. Supported models: {model_list}")

    # 打印模型大小，即参数量
    print(config.model_name, 'Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    # 加载预训练权重
    if pretrain is None:
        raise RuntimeError(f"need a pretrained model")
    else:
        if os.path.isfile(pretrain):
            print("=> 加载预训练权重 '{}'".format(pretrain))
            checkpoint = model.load_checkpoint(pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise RuntimeError(f"no pretrained model found at '{pretrain}'")


    return model





























