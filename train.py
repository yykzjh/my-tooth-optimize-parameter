import os
import nni


from collections import Counter
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D

import torch
from torch.utils.data import Dataset
from lib import utils, models, losses, augments



# 默认参数,这里的参数在后面添加到模型中，以params['dropout_rate']等替换原来的参数
params = {
"""
************************************************************************************************************************
————————————————————————————————————————————————     启动初始化    ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
    "CUDA_VISIBLE_DEVICES": "0",  # 选择可用的GPU编号

    "seed": 1777777,  # 随机种子

    "cuda": True,  # 是否使用GPU

    "benchmark": False,  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。用于网络输入维度和类型不会变的情况。

    "deterministic": True,  # 固定cuda的随机数种子，每次返回的卷积算法将是确定的。用于复现模型结果

"""
************************************************************************************************************************
————————————————————————————————————————————————     预处理       ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
    "resample_spacing": [0.5, 0.5, 0.5],  # 重采样的体素间距。三个维度的值一般相等，可设为0.5(图像尺寸有[200,200,100]、[200,200,200]、
    # [160,160,160]),或者设为0.25(图像尺寸有[400,400,200]、[400,400,400]、[320,320,320])

    "clip_lower_bound": 30,  # clip的下边界百分位点，图像中每个体素的数值按照从大到小排序，其中小于30%分位点的数值都等于30%分位数
    "clip_upper_bound": 99.9,  # clip的上边界百分位点，图像中每个体素的数值按照从大到小排序，其中大于100%分位点的数值都等于100%分位数

    "normalization": "full_volume_mean",  # 采用的归一化方法，可选["full_volume_mean","mean","max","max_min"]。其中"full_volume_mean"
    # 采用的是整个图像计算出的均值和标准差,"mean"采用的是图像前景计算出的均值和标准差,"max"是将图像所有数值除最大值,"max_min"采用的是Min-Max归一化

    "samples_train": 2048,  # 作为实际的训练集采样的子卷数量，也就是在原训练集上随机裁剪的子图像数量

    "crop_size": (160, 160, 96),  # 随机裁剪的尺寸。1、每个维度都是32的倍数这样在下采样时不会报错;2、11G的显存最大尺寸不能超过(192,192,160);
    # 3、要依据上面设置的"resample_spacing",在每个维度随机裁剪的尺寸不能超过图像重采样后的尺寸;

    "crop_threshold": 0.1,  # 随机裁剪时需要满足的条件，不满足则重新随机裁剪的位置。条件表示的是裁剪区域中的前景占原图总前景的最小比例

"""
************************************************************************************************************************
————————————————————————————————————————————————     数据增强    ————————————————————————————————————————————————————————
************************************************************************************************************************
"""
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

"""
************************************************************************************************************************
————————————————————————————————————————————————     数据读取     ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
    "dataset_name": "3DTooth",  # 数据集名称， 可选["3DTooth", ]

    "dataset_path": r"./datasets/src_10",  # 数据集路径

    "batch_size": 4,  # batch_size大小

    "num_workers": 4,  # num_workers大小

"""
************************************************************************************************************************
————————————————————————————————————————————————     网络模型      ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
    "model_name": "DENSEVNET",  # 模型名称，可选["DENSEVNET","VNET"]

    "in_channels": 1,  # 模型最开始输入的通道数,即模态数

    "classes": 35,  # 模型最后输出的通道数,即类别总数

"""
************************************************************************************************************************
————————————————————————————————————————————————      优化器      ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
    "optimizer_name": "adam",  # 优化器名称，可选["adam","sgd","rmsprop"]

    "learning_rate": 0.001,  # 学习率

    "weight_decay": 0.00001,  # 权重衰减系数,即更新网络参数时的L2正则化项的系数

    "momentum": 0.5,  # 动量大小

"""
************************************************************************************************************************
————————————————————————————————————————————————     损失函数     ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
    "loss_function_name": "DiceLoss",  # 损失函数名称，可选["DiceLoss","CrossEntropyLoss","WeightedCrossEntropyLoss","MSELoss",
    # "SmoothL1Loss","L1Loss","WeightedSmoothL1Loss","BCEDiceLoss","BCEWithLogitsLoss"]

    "class_weight": [0.1, 0.3, 3] + [1.0] * 32,  # 各类别计算损失值的加权权重

    "sigmoid_normalization": False,  # 对网络输出的各通道进行归一化的方式,True是对各元素进行sigmoid,False是对所有通道进行softmax

    "skip_index_after": None,  # 从某个索引的通道(类别)后不计算损失值

"""
************************************************************************************************************************
————————————————————————————————————————————————    训练相关参数   ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
    "runs_dir": r"./runs",  # 运行时产生的各类文件的存储根目录

    "start_epoch": 0,  # 训练时的起始epoch
    "end_epoch": 50,  # 训练时的结束epoch

    "best_dice": 0.60,  # 保存检查点的初始条件

    "terminal_show_freq": 50,  # 终端打印统计信息的频率,以step为单位

"""
************************************************************************************************************************
————————————————————————————————————————————————    测试相关参数   ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
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
        self.threshold = params["crop_threshold"]
        self.normalization = params["normalization"]
        self.augmentations = [
            params["open_elastic_transform"], params["open_gaussian_noise"], params["open_random_flip"],
            params["open_random_rescale"], params["open_random_rotate"], params["open_random_shift"]]
        self.list = []
        self.samples_train = params["samples_train"]

        # 定义数据增强
        all_transforms = [
            augments.ElasticTransform(alpha=params["elastic_transform_alpha"], sigma=params["elastic_transform_sigma"]),
            augments.GaussianNoise(mean=params["gaussian_noise_mean"], std=params["gaussian_noise_std"]),
            augments.RandomFlip(),
            augments.RandomRescale(min_percentage=params["random_rescale_min_percentage"],
                                   max_percentage=params["random_rescale_max_percentage"]),
            augments.RandomRotation(min_angle=params["random_rotate_min_angle"],
                                    max_angle=params["random_rotate_max_angle"]),
            augments.RandomShift(max_percentage=params["random_shift_max_percentage"])
        ],
        # 获取实际要进行的数据增强
        practice_transforms = [all_transforms[i] for i, is_open in enumerate(self.augmentations) if is_open]
        # 定义数据增强方式
        if params["augmentation_method"] == "Choice":
            self.transform = augments.RandomChoice(
                practice_transforms,
                p=params["augmentation_probability"]
            )
        elif params["augmentation_method"] == "Compose":
            self.transform = augments.ComposeTransforms(
                practice_transforms,
                p=params["augmentation_probability"]
            )


        # 定义子卷根目录
        sub_volume_root_path = os.path.join(
            self.root,
            "sub_volumes",
            self.mode +
            '-vol_' + str(config.crop_size[0]) + 'x' + str(config.crop_size[1]) + 'x' + str(config.crop_size[2]) +
            "-samples_" + str(self.samples)
        )
        # 定义子卷图像保存地址
        self.sub_vol_path = os.path.join(sub_volume_root_path, "generated")
        # 定义子卷图像路径保存的txt地址
        self.list_txt_path = os.path.join(sub_volume_root_path, "list.txt")

        # 直接加载之前生成的数据
        if load:
            self.list = utils.load_list(self.list_txt_path)
            return

        # 创建子卷根目录
        utils.make_dirs(sub_volume_root_path)
        # 创建子卷图像保存地址
        utils.make_dirs(self.sub_vol_path)

        # 分类创建子卷数据集
        if self.mode == 'train':
            images_path_list = sorted(glob.glob(os.path.join(self.train_path, "images", "*.nrrd")))
            labels_path_list = sorted(glob.glob(os.path.join(self.train_path, "labels", "*.nrrd")))

            self.list = create_sub_volumes(images_path_list, labels_path_list, samples=self.samples, sub_vol_path=self.sub_vol_path)

        elif self.mode == 'val':
            images_path_list = sorted(glob.glob(os.path.join(self.val_path, "images", "*.nrrd")))
            labels_path_list = sorted(glob.glob(os.path.join(self.val_path, "labels", "*.nrrd")))

            self.list = create_sub_volumes(images_path_list, labels_path_list, samples=self.samples, sub_vol_path=self.sub_vol_path)

            self.full_volume = get_viz_set(images_path_list, labels_path_list, image_index=0)

        elif self.mode == 'viz':
            images_path_list = sorted(glob.glob(os.path.join(self.val_path, "images", "*.nrrd")))
            labels_path_list = sorted(glob.glob(os.path.join(self.val_path, "labels", "*.nrrd")))

            self.full_volume = get_viz_set(images_path_list, labels_path_list, image_index=0)
            self.list = []

        # 保存所有子卷图像路径到txt文件
        utils.save_list(self.list_txt_path, self.list)


    def __len__(self):
        return len(self.list)


    def __getitem__(self, index):
        image_path, label_path = self.list[index]
        image, label = np.load(image_path), np.load(label_path)

        if self.mode == 'train' and self.augmentation:
            augmented_image, augmented_label = self.transform(image, label)

            return torch.FloatTensor(augmented_image.copy()).unsqueeze(0), torch.FloatTensor(augmented_label.copy())

        return torch.FloatTensor(image).unsqueeze(0), torch.FloatTensor(label)











if __name__ == '__main__':

    # 获得下一组搜索空间中的参数
    tuner_params = nni.get_next_parameter()
    # 更新参数
    params.update(tuner_params)

    # 设置可用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    # 随机种子、卷积算法优化
    utils.reproducibility(params["seed"], params["cuda"], params["deterministic"], params["benchmark"])




