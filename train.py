import os
import nni
import glob
import shutil
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lib import utils, transforms



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

    "clip_lower_bound": 30,  # clip的下边界百分位点，图像中每个体素的数值按照从大到小排序，其中小于30%分位点的数值都等于30%分位数
    "clip_upper_bound": 99.9,  # clip的上边界百分位点，图像中每个体素的数值按照从大到小排序，其中大于100%分位点的数值都等于100%分位数

    "samples_train": 256,  # 作为实际的训练集采样的子卷数量，也就是在原训练集上随机裁剪的子图像数量

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
    "normalize_mean": 0.5,
    "normalize_std": 1.5,

    # —————————————————————————————————————————————    数据读取     ——————————————————————————————————————————————————————

    "dataset_name": "3DTooth",  # 数据集名称， 可选["3DTooth", ]

    "dataset_path": r"./datasets/src_10",  # 数据集路径

    "batch_size": 4,  # batch_size大小

    "num_workers": 4,  # num_workers大小

    # —————————————————————————————————————————————    网络模型     ——————————————————————————————————————————————————————

    "model_name": "DENSEVNET",  # 模型名称，可选["DENSEVNET","VNET"]

    "in_channels": 1,  # 模型最开始输入的通道数,即模态数

    "classes": 35,  # 模型最后输出的通道数,即类别总数

    # ——————————————————————————————————————————————    优化器     ——————————————————————————————————————————————————————

    "optimizer_name": "adam",  # 优化器名称，可选["adam","sgd","rmsprop"]

    "learning_rate": 0.001,  # 学习率

    "weight_decay": 0.00001,  # 权重衰减系数,即更新网络参数时的L2正则化项的系数

    "momentum": 0.5,  # 动量大小

    # ————————————————————————————————————————————    损失函数     ———————————————————————————————————————————————————————

    "loss_function_name": "DiceLoss",  # 损失函数名称，可选["DiceLoss","CrossEntropyLoss","WeightedCrossEntropyLoss",
    # "MSELoss","SmoothL1Loss","L1Loss","WeightedSmoothL1Loss","BCEDiceLoss","BCEWithLogitsLoss"]

    "class_weight": [0.1, 0.3, 3] + [1.0] * 32,  # 各类别计算损失值的加权权重

    "sigmoid_normalization": False,  # 对网络输出的各通道进行归一化的方式,True是对各元素进行sigmoid,False是对所有通道进行softmax

    "skip_index_after": None,  # 从某个索引的通道(类别)后不计算损失值

    # —————————————————————————————————————————————   训练相关参数   ——————————————————————————————————————————————————————

    "runs_dir": r"./runs",  # 运行时产生的各类文件的存储根目录

    "start_epoch": 0,  # 训练时的起始epoch
    "end_epoch": 50,  # 训练时的结束epoch

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
                    transforms.ToTensor(),
                    transforms.Normalize(params["normalize_mean"], params["normalize_std"])
                ])
            elif params["augmentation_method"] == "Compose":
                self.train_transforms = transforms.ComposeTransforms([
                    transforms.ComposeAugments(practice_augments, p=params["augmentation_probability"]),
                    transforms.ToTensor(),
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
                transforms.ToTensor(),
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









if __name__ == '__main__':

    # # 获得下一组搜索空间中的参数
    # tuner_params = nni.get_next_parameter()
    # # 更新参数
    # params.update(tuner_params)


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

    for images, labels in train_loader:
        OrthoSlicer3D(images[0, 0, :, :, :].numpy()).show()
        OrthoSlicer3D(labels[0, :, :, :].numpy()).show()



