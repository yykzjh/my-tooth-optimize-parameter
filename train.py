import os
import glob
import math
import tqdm
import shutil
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import nni
from nni.experiment import Experiment

from lib import utils, transforms, models, losses



# 默认参数,这里的参数在后面添加到模型中，以params['dropout_rate']等替换原来的参数
params = {
    # ——————————————————————————————————————————————     启动初始化    ———————————————————————————————————————————————————

    "CUDA_VISIBLE_DEVICES": "0",  # 选择可用的GPU编号

    "seed": 1777777,  # 随机种子

    "cuda": True,  # 是否使用GPU

    "benchmark": False,  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。用于网络输入维度和类型不会变的情况。

    "deterministic": True,  # 固定cuda的随机数种子，每次返回的卷积算法将是确定的。用于复现模型结果

    # —————————————————————————————————————————————     预处理       ————————————————————————————————————————————————————

    "resample_spacing": [0.5, 0.5, 0.5],  # 重采样的体素间距。三个维度的值一般相等，可设为0.5(图像尺寸有[200,200,100]、[200,200,200]、
    # [160,160,160]),或者设为0.25(图像尺寸有[400,400,200]、[400,400,400]、[320,320,320])

    "clip_lower_bound": -3566,  # clip的下边界数值
    "clip_upper_bound": 14913,  # clip的上边界数值

    "samples_train": 2048,  # 作为实际的训练集采样的子卷数量，也就是在原训练集上随机裁剪的子图像数量

    "crop_size": (160, 160, 96),  # 随机裁剪的尺寸。1、每个维度都是32的倍数这样在下采样时不会报错;2、11G的显存最大尺寸不能超过(192,192,160);
    # 3、要依据上面设置的"resample_spacing",在每个维度随机裁剪的尺寸不能超过图像重采样后的尺寸;

    "crop_threshold": 0.1,  # 随机裁剪时需要满足的条件，不满足则重新随机裁剪的位置。条件表示的是裁剪区域中的前景占原图总前景的最小比例

    # ——————————————————————————————————————————————    数据增强    ——————————————————————————————————————————————————————

    "augmentation_probability": 0.3,  # 每张图像做数据增强的概率
    "augmentation_method": "Choice",  # 数据增强的方式，可选["Compose", "Choice"]

    # 弹性形变参数
    "open_elastic_transform": True,  # 是否开启弹性形变数据增强
    "elastic_transform_sigma": 20,  # 高斯滤波的σ,值越大，弹性形变越平滑
    "elastic_transform_alpha": 1,  # 形变的幅度，值越大，弹性形变越剧烈

    # 高斯噪声参数
    "open_gaussian_noise": True,  # 是否开启添加高斯噪声数据增强
    "gaussian_noise_mean": 0,  # 高斯噪声分布的均值
    "gaussian_noise_std": 0.01,  # 高斯噪声分布的标准差,值越大噪声越强

    # 随机翻转参数
    "open_random_flip": True,  # 是否开启随机翻转数据增强

    # 随机缩放参数
    "open_random_rescale": True,  # 是否开启随机缩放数据增强
    "random_rescale_min_percentage": 0.5,  # 随机缩放时,最小的缩小比例
    "random_rescale_max_percentage": 1.5,  # 随机缩放时,最大的放大比例

    # 随机旋转参数
    "open_random_rotate": True,  # 是否开启随机旋转数据增强
    "random_rotate_min_angle": -50,  # 随机旋转时,反方向旋转的最大角度
    "random_rotate_max_angle": 50,  # 随机旋转时,正方向旋转的最大角度

    # 随机位移参数
    "open_random_shift": True,  # 是否开启随机位移数据增强
    "random_shift_max_percentage": 0.3,  # 在图像的三个维度(D,H,W)都进行随机位移，位移量的范围为(-0.3×(D、H、W),0.3×(D、H、W))

    # 标准化均值
    "normalize_mean": 0.17375025153160095,
    "normalize_std": 0.053983770310878754,

    # —————————————————————————————————————————————    数据读取     ——————————————————————————————————————————————————————

    "dataset_name": "3DTooth",  # 数据集名称， 可选["3DTooth", ]

    "dataset_path": r"./datasets/src_10",  # 数据集路径

    "batch_size": 1,  # batch_size大小

    "num_workers": 1,  # num_workers大小

    # —————————————————————————————————————————————    网络模型     ——————————————————————————————————————————————————————

    "model_name": "DENSEVNET",  # 模型名称，可选["DENSEVNET","VNET"]

    "in_channels": 1,  # 模型最开始输入的通道数,即模态数

    "classes": 35,  # 模型最后输出的通道数,即类别总数

    # ——————————————————————————————————————————————    优化器     ——————————————————————————————————————————————————————

    "optimizer_name": "Adam",  # 优化器名称，可选["SGD", "Adagrad", "RMSprop", "Adam", "Adamax", "Adadelta", "SparseAdam"]

    "learning_rate": 0.001,  # 学习率

    "weight_decay": 0.00001,  # 权重衰减系数,即更新网络参数时的L2正则化项的系数

    "momentum": 0.9,  # 动量大小

    # ———————————————————————————————————————————    学习率调度器     —————————————————————————————————————————————————————

    "lr_scheduler_name": "ReduceLROnPlateau",  # 学习率调度器名称，可选["ExponentialLR", "StepLR", "MultiStepLR",
    # "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "OneCycleLR", "ReduceLROnPlateau"]

    "gamma": 0.9,  # 学习率衰减系数

    "step_size": 5,  # StepLR的学习率衰减步长

    "milestones": [15, 22, 27, 30, 33, 36, 38],  # MultiStepLR的学习率衰减节点列表

    "T_max": 5,  # CosineAnnealingLR的半周期

    "T_0": 4,  # CosineAnnealingWarmRestarts的周期

    "T_mult": 2,  # CosineAnnealingWarmRestarts的周期放大倍数

    "mode": "min",  # ReduceLROnPlateau的衡量指标变化方向

    "patience": 5,  # ReduceLROnPlateau的衡量指标可以停止优化的最长epoch

    "factor": 0.1,  # ReduceLROnPlateau的衰减系数

    # ————————————————————————————————————————————    损失函数     ———————————————————————————————————————————————————————

    "loss_function_name": "DiceLoss",  # 损失函数名称，可选["DiceLoss","CrossEntropyLoss","WeightedCrossEntropyLoss",
    # "MSELoss","SmoothL1Loss","L1Loss","WeightedSmoothL1Loss","BCEDiceLoss","BCEWithLogitsLoss"]

    "class_weight": [0.00002066, 0.00022885, 0.10644896, 0.02451709, 0.03155127, 0.02142642, 0.02350483, 0.02480525,
                     0.01125384, 0.01206108, 0.07426875, 0.02583742, 0.03059388, 0.02485595, 0.02466779, 0.02529981,
                     0.01197175, 0.01272877, 0.16020726, 0.05647514, 0.0285633, 0.01808694, 0.02124704, 0.02175892,
                     0.01802092, 0.01563035, 0., 0.0555509, 0.02747846, 0.01756969, 0.02183707, 0.01934677, 0.01848419,
                     0.01370064, 0.],  # 各类别计算损失值的加权权重

    "sigmoid_normalization": False,  # 对网络输出的各通道进行归一化的方式,True是对各元素进行sigmoid,False是对所有通道进行softmax

    # —————————————————————————————————————————————   训练相关参数   ——————————————————————————————————————————————————————

    "run_dir": r"./runs",  # 运行时产生的各类文件的存储根目录

    "start_epoch": 1,  # 训练时的起始epoch
    "end_epoch": 40,  # 训练时的结束epoch

    "best_dice": 0.60,  # 保存检查点的初始条件

    "terminal_show_freq": 50,  # 终端打印统计信息的频率,以step为单位

    # ————————————————————————————————————————————   测试相关参数   ———————————————————————————————————————————————————————

    "crop_stride": [4, 4, 4]
}




class ToothDataset(Dataset):
    """
    读取nrrd牙齿数据集
    """

    def __init__(self, mode):
        """
        Args:
            mode: train/val
        """
        self.mode = mode
        self.run_dir = params["run_dir"]
        self.root = params["dataset_path"]
        self.train_path = os.path.join(self.root, "train")
        self.val_path = os.path.join(self.root, "val")
        self.augmentations = [
            params["open_elastic_transform"], params["open_gaussian_noise"], params["open_random_flip"],
            params["open_random_rescale"], params["open_random_rotate"], params["open_random_shift"]]
        self.sub_volume_root_dir = os.path.join(self.root, "sub_volumes")
        self.data = []
        self.transform = None

        # 分类创建子卷数据集
        if self.mode == 'train':
            # 定义数据增强
            all_augments = [
                 transforms.ElasticTransform(alpha=params["elastic_transform_alpha"],
                                             sigma=params["elastic_transform_sigma"]),
                 transforms.GaussianNoise(mean=params["gaussian_noise_mean"],
                                          std=params["gaussian_noise_std"]),
                 transforms.RandomFlip(),
                 transforms.RandomRescale(min_percentage=params["random_rescale_min_percentage"],
                                          max_percentage=params["random_rescale_max_percentage"]),
                 transforms.RandomRotation(min_angle=params["random_rotate_min_angle"],
                                           max_angle=params["random_rotate_max_angle"]),
                 transforms.RandomShift(max_percentage=params["random_shift_max_percentage"])
            ]
            # 获取实际要进行的数据增强
            practice_augments = [all_augments[i] for i, is_open in enumerate(self.augmentations) if is_open]
            # 定义数据增强方式
            if params["augmentation_method"] == "Choice":
                self.train_transforms = transforms.ComposeTransforms([
                    transforms.RandomAugmentChoice(practice_augments, p=params["augmentation_probability"]),
                    transforms.ToTensor(params["clip_lower_bound"], params["clip_upper_bound"]),
                    # transforms.ToTensor(),
                    transforms.Normalize(params["normalize_mean"], params["normalize_std"])
                ])
            elif params["augmentation_method"] == "Compose":
                self.train_transforms = transforms.ComposeTransforms([
                    transforms.ComposeAugments(practice_augments, p=params["augmentation_probability"]),
                    transforms.ToTensor(params["clip_lower_bound"], params["clip_upper_bound"]),
                    # transforms.ToTensor(),
                    transforms.Normalize(params["normalize_mean"], params["normalize_std"])
                ])

            # 定义子卷训练集目录
            self.sub_volume_train_dir = os.path.join(self.sub_volume_root_dir, "train")
            # 定义子卷原图像存储目录
            self.sub_volume_images_dir = os.path.join(self.sub_volume_train_dir, "images")
            # 定义子卷标注图像存储目录
            self.sub_volume_labels_dir = os.path.join(self.sub_volume_train_dir, "labels")

            # 创建子卷训练集目录
            utils.make_dirs(self.sub_volume_train_dir)
            # 创建子卷原图像存储目录
            utils.make_dirs(self.sub_volume_images_dir)
            # 创建子卷标注图像存储目录
            utils.make_dirs(self.sub_volume_labels_dir)

            # 获取数据集中所有原图图像和标注图像的路径
            images_path_list = sorted(glob.glob(os.path.join(self.train_path, "images", "*.nrrd")))
            labels_path_list = sorted(glob.glob(os.path.join(self.train_path, "labels", "*.nrrd")))

            # 生成子卷数据集
            self.data = utils.create_sub_volumes(images_path_list, labels_path_list, params["samples_train"],
                                                 params["resample_spacing"], params["clip_lower_bound"],
                                                 params["clip_upper_bound"], params["crop_size"],
                                                 params["crop_threshold"], self.sub_volume_train_dir)

        elif self.mode == 'val':
            # 定义验证集数据增强
            self.val_transforms = transforms.ComposeTransforms([
                transforms.ToTensor(params["clip_lower_bound"], params["clip_upper_bound"]),
                # transforms.ToTensor(),
                transforms.Normalize(params["normalize_mean"], params["normalize_std"])
            ])

            # 定义子卷验证集目录
            self.sub_volume_val_dir = os.path.join(self.sub_volume_root_dir, "val")
            # 定义子卷原图像存储目录
            self.sub_volume_images_dir = os.path.join(self.sub_volume_val_dir, "images")
            # 定义子卷标注图像存储目录
            self.sub_volume_labels_dir = os.path.join(self.sub_volume_val_dir, "labels")

            # 创建子卷训练集目录
            utils.make_dirs(self.sub_volume_val_dir)
            # 创建子卷原图像存储目录
            utils.make_dirs(self.sub_volume_images_dir)
            # 创建子卷标注图像存储目录
            utils.make_dirs(self.sub_volume_labels_dir)

            # 获取数据集中所有原图图像和标注图像的路径
            images_path_list = sorted(glob.glob(os.path.join(self.val_path, "images", "*.nrrd")))
            labels_path_list = sorted(glob.glob(os.path.join(self.val_path, "labels", "*.nrrd")))

            # 对验证集进行预处理
            self.data = utils.preprocess_val_dataset(images_path_list, labels_path_list, params["resample_spacing"],
                                                     params["clip_lower_bound"], params["clip_upper_bound"],
                                                     self.sub_volume_val_dir)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        image_path, label_path = self.data[index]
        image, label = np.load(image_path), np.load(label_path)

        if self.mode == 'train':
            transform_image, transform_label = self.train_transforms(image, label)
            return transform_image.unsqueeze(0), transform_label

        else:
            transform_image, transform_label = self.val_transforms(image, label)
            return transform_image.unsqueeze(0), transform_label




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




def split_forward(image, model):
    """
    对于验证集完整图像，需要滑动切块后分别进行预测，最后再拼接到一起

    Args:
        image: 验证集完整图像
        model: 网络模型

    Returns:

    """
    # 获取图像尺寸
    ori_shape = image.size()[2:]
    # 初始化输出的特征图
    output = torch.zeros((image.size()[0], params["classes"], *ori_shape), device=image.device)
    # 切片的大小
    slice_shape = params["crop_size"]
    # 在三个维度上滑动的步长
    stride = params["crop_stride"]

    # 在三个维度上进行滑动切片
    for shape0_start in range(0, ori_shape[0], stride[0]):
        shape0_end = shape0_start + slice_shape[0]
        start0 = shape0_start
        end0 = shape0_end
        if shape0_end >= ori_shape[0]:
            end0 = ori_shape[0]
            start0 = end0 - slice_shape[0]

        for shape1_start in range(0, ori_shape[1], stride[1]):
            shape1_end = shape1_start + slice_shape[1]
            start1 = shape1_start
            end1 = shape1_end
            if shape1_end >= ori_shape[1]:
                end1 = ori_shape[1]
                start1 = end1 - slice_shape[1]

            for shape2_start in range(0, ori_shape[2], stride[2]):
                shape2_end = shape2_start + slice_shape[2]
                start2 = shape2_start
                end2 = shape2_end
                if shape2_end >= ori_shape[2]:
                    end2 = ori_shape[2]
                    start2 = end2 - slice_shape[2]

                slice_tensor = image[:, :, start0:end0, start1:end1, start2:end2]
                slice_predict = model(slice_tensor.to(image.device))
                output[:, :, start0:end0, start1:end1, start2:end2] += slice_predict

                if shape2_end >= ori_shape[2]:
                    break

            if shape1_end >= ori_shape[1]:
                break

        if shape0_end >= ori_shape[0]:
            break

    return output






if __name__ == '__main__':

    # 获得下一组搜索空间中的参数
    tuner_params = nni.get_next_parameter()
    # 更新参数
    params.update(tuner_params)


    # 设置可用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    # 随机种子、卷积算法优化
    utils.reproducibility(params["seed"], params["cuda"], params["deterministic"], params["benchmark"])


    # 初始化数据集
    train_set = ToothDataset(mode="train")
    val_set = ToothDataset(mode="val")


    # 初始化数据加载器
    train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True,
                              num_workers=params["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


    # 初始化网络模型
    if params["model_name"] == 'DENSEVNET':
        model = models.DenseVNet(in_channels=params["in_channels"], classes=params["classes"])

    else:
        raise RuntimeError(f"{params['model_name']}是不支持的网络模型！")

    # 随机初始化模型参数
    model.apply(weight_init)


    # 初始化优化器
    if params["optimizer_name"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"], momentum=params["momentum"],
                              weight_decay=params["weight_decay"])

    elif params["optimizer_name"] == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

    elif params["optimizer_name"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"],
                                  momentum=params["momentum"])

    elif params["optimizer_name"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

    elif params["optimizer_name"] == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

    elif params["optimizer_name"] == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

    elif params["optimizer_name"] == "SparseAdam":
        optimizer = optim.SparseAdam(model.parameters(), lr=params["learning_rate"])

    else:
        raise RuntimeError(
            f"{params['optimizer_name']}是不支持的优化器！")


    # 初始化学习率调度器
    if params["lr_scheduler_name"] == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["gamma"])

    elif params["lr_scheduler_name"] == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])

    elif params["lr_scheduler_name"] == "MultiStepLR":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params["milestones"], gamma=params["gamma"])

    elif params["lr_scheduler_name"] == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["T_max"])

    elif params["lr_scheduler_name"] == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=params["T_0"],
                                                                      T_mult=params["T_mult"])

    elif params["lr_scheduler_name"] == "OneCycleLR":
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params["learning_rate"],
                                                     steps_per_epoch=len(train_loader), epochs=params["end_epoch"])

    elif params["lr_scheduler_name"] == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=params["mode"], factor=params["factor"],
                                                            patience=params["patience"])
    else:
        raise RuntimeError(
            f"{params['lr_scheduler_name']}是不支持的学习率调度器！")


    # 把模型放到GPU上
    if params["cuda"]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    else:
        device = torch.device("cpu")


    # 初始化损失函数
    if params["loss_function_name"] == "DiceLoss":
        loss_function = losses.DiceLoss(params["classes"], weight=torch.FloatTensor(params["class_weight"]).to(device),
                                        sigmoid_normalization=False)

    else:
        raise RuntimeError(
            f"{params['loss_function_name']}是不支持的损失函数！")


    # 初始化在验证集上的最优DSC
    val_best_dsc = 0.0

    # 开始训练
    for epoch in range(params["start_epoch"], params["end_epoch"] + 1):
        # 初始化当前epoch所有训练集图像的loss总和
        train_loss_sum_per_epoch = 0.0

        # 训练
        model.train()
        # 遍历数据集的batch
        for batch_idx, (input_tensor, target) in enumerate(train_loader):
            # 梯度清0
            optimizer.zero_grad()
            # 将输入图像和标注图像都移动到指定设备上
            input_tensor, target = input_tensor.to(device), target.to(device)
            # 前向传播
            output = model(input_tensor)
            # 计算损失值
            dice_loss = loss_function(output, target)
            # 将当前loss累加到loss总和
            train_loss_sum_per_epoch += dice_loss.item() * input_tensor.size(0)
            # 反向传播计算各参数的梯度
            dice_loss.backward()
            # 更新参数
            optimizer.step()

        # 计算当前epoch训练集图像的平均loss
        train_loss_mean_per_epoch = train_loss_sum_per_epoch / len(train_set)
        # 更新学习率
        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(train_loss_mean_per_epoch)
        else:
            lr_scheduler.step()


        # 初始化当前epoch所有验证集图像的dsc总和
        val_dsc_sum_per_epoch = 0.0

        # 验证集测试
        model.eval()
        # 测试时不保存计算图的梯度中间结果，加快速度，节省空间
        with torch.no_grad():
            # 遍历验证集的batch，默认一个batch一张图像
            for batch_idx, (input_tensor, target) in enumerate(val_loader):
                # 将输入图像和标注图像都移动到指定设备上
                input_tensor, target = input_tensor.to(device), target.to(device)
                # 前向传播
                output = split_forward(input_tensor, model)
                # 计算每个类别的dsc
                per_class_dsc = loss_function.dice(output, target, mode="standard")
                # 将所有类别的dsc计算平均值，然后累加到dsc总和
                val_dsc_sum_per_epoch += per_class_dsc.mean().item() * input_tensor.size(0)

        # 计算当前epoch验证集图像的平均dsc
        val_dsc_mean_per_epoch = val_dsc_sum_per_epoch / len(val_set)
        # 向nni上报每个epoch验证集的平均dsc作为中间指标
        nni.report_intermediate_result(val_dsc_mean_per_epoch)
        # 更新在验证集上的最优dsc
        val_best_dsc = max(val_best_dsc, val_dsc_mean_per_epoch)

        # 打印中间结果
        print("epoch:[{:02d}/{:02d}]   mean_train_loss:{:.06f}   mean_val_dsc:{:.06f}   val_best_dsc:{:.06f}"
              .format(epoch, params["end_epoch"], train_loss_mean_per_epoch, val_dsc_mean_per_epoch, val_best_dsc))

    # 将在验证集上最优的dsc作为最终上报指标
    nni.report_final_result(val_best_dsc)

    # 实验结束后，防止程序避免Python解释器自动退出，可以继续使用网页控制台
    input()



