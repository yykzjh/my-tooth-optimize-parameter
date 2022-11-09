import os

import numpy as np
import scipy
import nrrd
import glob
import math
import torch
import nibabel as nib
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from lib import utils




def show_space_property(dataset_path=r"./datasets/src_10"):
    train_images = glob.glob(os.path.join(dataset_path, 'train', "images", '*.nrrd'))
    train_labels = glob.glob(os.path.join(dataset_path, 'train', "labels", '*.nrrd'))
    val_images = glob.glob(os.path.join(dataset_path, 'val', "images", '*.nrrd'))
    val_labels = glob.glob(os.path.join(dataset_path, 'val', "labels", '*.nrrd'))
    keys = ["type", "dimension", "space", "sizes", "space directions", "space origin"]
    values = []
    for i, key in enumerate(keys):
        values.append([])
        for path in train_images:
            # print(path)
            _, options = nrrd.read(path)
            values[i].append(options[key])
        for path in train_labels:
            # print(path)
            _, options = nrrd.read(path)
            values[i].append(options[key])
        for path in val_images:
            # print(path)
            _, options = nrrd.read(path)
            values[i].append(options[key])
        for path in val_labels:
            # print(path)
            _, options = nrrd.read(path)
            values[i].append(options[key])
        print(key + ": ", values[i])



def analyse_dataset(dataset_path=r"./datasets/src_10"):
    # 加载数据集中所有标注图像
    labels_path = glob.glob(os.path.join(dataset_path, "*", "labels", '*.nrrd'))
    # 对所有标注图像进行重采样，统一spacing
    labels = [utils.load_image_or_label(label_path, [0.5, 0.5, 0.5], 30, 99.9, type="label")
              for label_path in labels_path]

    # 加载类别名称和类别索引映射字典
    # 读取索引文件
    index_to_class_dict = utils.load_json_file(r"./3DTooth.json")
    class_to_index_dict = {}
    # 获得类别到索引的映射字典
    for key, val in index_to_class_dict.items():
        class_to_index_dict[val] = key

    # 初始化类别统计字典
    class_statistic_dict = {class_index: {
        "tooth_num": 0,
        "voxel_num": 0
    } for class_index in range(35)}
    # 初始化每张图像的统计字典
    label_statistic_dict = {os.path.splitext(os.path.basename(label_path))[0]: {
        "tooth_num": 0,
        "have_implant": False,
        "total_voxel_num": 0,
        "foreground_voxel_num": 0,
        "slice_num": 0
    } for label_path in labels_path}

    # 遍历所有标注图像
    for label_i, label in enumerate(labels):
        # 获得当前标注图像的名称
        label_name = os.path.splitext(os.path.basename(labels_path[label_i]))[0]
        # 获取当前标注图像的统计数据
        class_indexes, indexes_cnt = torch.unique(label, return_counts=True)
        # 遍历类别索引，将统计数据累加到类别统计字典
        for i, _ in enumerate(class_indexes):
            class_index = class_indexes[i].item()
            class_statistic_dict[class_index]["tooth_num"] += 1  # 类别计数加1
            class_statistic_dict[class_index]["voxel_num"] += indexes_cnt[i].item()  # 类别体素数累加
        # 更新每张标注图像的统计字典
        label_statistic_dict[label_name]["tooth_num"] = torch.nonzero(class_indexes > 2).shape[0]
        label_statistic_dict[label_name]["have_implant"] = True if 2 in class_indexes else False
        label_statistic_dict[label_name]["total_voxel_num"] = label.numel()
        label_statistic_dict[label_name]["foreground_voxel_num"] = torch.nonzero(label != 0).shape[0]
        label_statistic_dict[label_name]["slice_num"] = label.shape[2]
    # 对两个统计字典按键排序
    class_statistic_dict = dict(sorted(class_statistic_dict.items(), key=lambda x: x[0]))
    label_statistic_dict = dict(sorted(label_statistic_dict.items(), key=lambda x: x[0]))

    # 设置解决中文乱码问题
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    # 定义直方图条形宽度
    width = 0.8



    # 获得每类牙齿的统计数量
    tooth_num = [sub_dict["tooth_num"] for class_index, sub_dict in class_statistic_dict.items() if class_index != 0]
    # 展示各类牙齿的数量对比直方图
    plt.bar([i for i in range(1, 35)],
            tooth_num,
            width=width,
            tick_label=[index_to_class_dict[i] for i in range(1, 35)],
            label='牙齿数量')
    # 设置y轴显示范围
    plt.ylim([0, max(tooth_num) * 1.2])
    # 为每个条形图添加数值标签
    for i, num in enumerate(tooth_num):
        plt.text(i + 1, num, num, ha='center', fontsize=12)
    # 设置图例
    plt.legend()
    # 设置标题
    plt.title('不同种类的牙齿数量对比')
    # 设置轴上的标题
    plt.xlabel("牙齿类别")
    plt.ylabel("数量(颗)")
    plt.show()



    # 获得每类牙齿的体素的统计数量
    voxel_num = [sub_dict["voxel_num"] for class_index, sub_dict in class_statistic_dict.items() if class_index != 0]
    # 展示各类牙齿的体素数量对比直方图
    plt.bar([i for i in range(1, 35)],
            voxel_num,
            width=width,
            tick_label=[index_to_class_dict[i] for i in range(1, 35)],
            label='牙齿体素数量')
    # 设置y轴显示范围
    plt.ylim([0, max(voxel_num) * 1.2])
    # 为每个条形图添加数值标签
    for i, num in enumerate(voxel_num):
        plt.text(i + 1, num, num, ha='center', fontsize=12)
    # 设置图例
    plt.legend()
    # 设置标题
    plt.title('不同种类的牙齿体素数量对比')
    # 设置轴上的标题
    plt.xlabel("牙齿类别")
    plt.ylabel("体素数量")
    plt.show()


    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16


    # 获得每张图像的牙齿统计数量
    tooth_cnt = [sub_dict["tooth_num"] for file_name, sub_dict in label_statistic_dict.items()]
    # 获得每张图像是否存在种植体
    exist_implant = [sub_dict["have_implant"] for file_name, sub_dict in label_statistic_dict.items()]
    # 获得每个条形设置的颜色
    bar_colors = ["r" if exist else "b" for exist in exist_implant]
    # 获取文件名称列表
    file_names = [file_name for file_name, sub_dict in label_statistic_dict.items()]
    # 展示每张图像牙齿数量对比直方图
    plt.bar([i for i in range(len(labels))],
            tooth_cnt,
            width=width,
            tick_label=file_names,
            color=bar_colors,
            label='牙齿数量')
    # 设置y轴显示范围
    plt.ylim([0, max(tooth_cnt) * 1.2])
    # 为每个条形图添加数值标签
    for i, num in enumerate(tooth_cnt):
        plt.text(i, num, num, ha='center', fontsize=16)
    # 设置图例
    plt.legend()
    # 设置标题
    plt.title('每张图像的牙齿数量对比')
    # 设置轴上的标题
    plt.xlabel("文件名称")
    plt.ylabel("数量(颗)")
    plt.show()



    # 重新设置宽度
    width = 0.4
    # 获得每张图像的前景体素统计数量
    foreground_voxel_num = [sub_dict["foreground_voxel_num"] for file_name, sub_dict in label_statistic_dict.items()]
    # 获得每张图像的总体素统计数量
    total_voxel_num = [sub_dict["total_voxel_num"] for file_name, sub_dict in label_statistic_dict.items()]
    # 获取文件名称列表
    file_names = [file_name for file_name, sub_dict in label_statistic_dict.items()]
    # 展示每张图像前景体素对比直方图
    plt.bar([i - width / 2 for i in range(len(labels))],
            foreground_voxel_num,
            width=width,
            tick_label=file_names,
            color="r",
            label='前景体素数量')
    # 展示每张图像总体素对比直方图
    plt.bar([i + width / 2 for i in range(len(labels))],
            total_voxel_num,
            width=width,
            tick_label=file_names,
            color="b",
            label='总体素数量')
    # 设置y轴显示范围
    plt.ylim([0, max(total_voxel_num) * 1.2])
    # 为每个条形图添加数值标签
    for i, num in enumerate(foreground_voxel_num):
        plt.text(i - width / 2, num, num, ha='center', fontsize=16)
    for i, num in enumerate(total_voxel_num):
        plt.text(i + width / 2, num, num, ha='center', fontsize=16)
    # 设置图例
    plt.legend()
    # 设置标题
    plt.title('每张图像的前景体素和总体素数量对比')
    # 设置轴上的标题
    plt.xlabel("文件名称")
    plt.ylabel("体素数量")
    plt.show()



    # 重新设置宽度
    width = 0.8
    # 获得每张图像的slice统计数量
    slice_num = [sub_dict["slice_num"] for file_name, sub_dict in label_statistic_dict.items()]
    # 获取文件名称列表
    file_names = [file_name for file_name, sub_dict in label_statistic_dict.items()]
    # 展示每张图像前景体素对比直方图
    plt.bar([i for i in range(len(labels))],
            slice_num,
            width=width,
            tick_label=file_names,
            label='切片slice数量')
    # 设置y轴显示范围
    plt.ylim([0, max(slice_num) * 1.2])
    # 为每个条形图添加数值标签
    for i, num in enumerate(slice_num):
        plt.text(i, num, num, ha='center', fontsize=16)
    # 设置图例
    plt.legend()
    # 设置标题
    plt.title('每张图像的slice切片数量对比')
    # 设置轴上的标题
    plt.xlabel("文件名称")
    plt.ylabel("切片数量")
    plt.show()


    print(class_statistic_dict)
    print(label_statistic_dict)



def calc_clip_bound(dataset_path=r"./datasets/src_10"):
    # 加载数据集中所有原图像
    images_path = glob.glob(os.path.join(dataset_path, "*", "images", '*.nrrd'))
    # 加载数据集中所有标注图像
    labels_path = glob.glob(os.path.join(dataset_path, "*", "labels", '*.nrrd'))

    # 对所有原图像进行重采样、clip和灰度值对齐
    images = [utils.load_image_or_label(image_path, [0.5, 0.5, 0.5], 30, 99.9, type="ori_image")
              for image_path in images_path]
    # 对所有标注图像进行重采样，统一spacing
    labels = [utils.load_image_or_label(label_path, [0.5, 0.5, 0.5], 30, 99.9, type="label")
              for label_path in labels_path]
    assert len(images) == len(labels), "原图像数量和标注图像数量不一致！"

    # 初始化数据结构
    all_values = torch.IntTensor([])
    foreground_values = torch.IntTensor([])
    # 遍历所有图像
    for i in range(len(images)):
        # 获取当前图像的原图像和标注图像
        image = images[i]
        label = labels[i]
        # 累计所有灰度值
        all_values = torch.cat([all_values, image.flatten()], dim=0)
        # 累计前景灰度值
        foreground_values = torch.cat([foreground_values, image[label.nonzero(as_tuple=True)]], dim=0)

    # 从小到大排序
    foreground_values = torch.sort(foreground_values)[0]
    # tensor转换成numpy.ndarray
    foreground_values_np = foreground_values.numpy()
    # 画直方图
    plt.hist(foreground_values_np, bins=int(foreground_values_np.max() - foreground_values_np.min() + 1))
    plt.axvline(foreground_values_np[int(0.004 * len(foreground_values_np))], color="r")
    plt.axvline(foreground_values_np[int(0.996 * len(foreground_values_np))], color="r")
    plt.show()

    # 输出clip的上下边界的范围,lower_bound:[all_min_val, foreground_min_val], upper_bound:[foreground_max_val, all_max_val]
    print("lower_bound_range: [{0}, {1}]".format(all_values.min(), foreground_values_np[int(0.004 * len(foreground_values_np))]))
    print("upper_bound_range: [{0}, {1}]".format(foreground_values_np[int(0.996 * len(foreground_values_np))], all_values.max()))




def calc_mean_and_std(dataset_path=r"./datasets/src_10"):
    # 加载数据集中所有原图像
    images_path = glob.glob(os.path.join(dataset_path, "*", "images", '*.nrrd'))
    # 加载数据集中所有标注图像
    labels_path = glob.glob(os.path.join(dataset_path, "*", "labels", '*.nrrd'))

    # 对所有原图像进行重采样、clip和灰度值对齐
    images = [utils.load_image_or_label(image_path, [0.5, 0.5, 0.5], 30, 99.9, type="ori_image")
              for image_path in images_path]
    # 对所有标注图像进行重采样，统一spacing
    labels = [utils.load_image_or_label(label_path, [0.5, 0.5, 0.5], 30, 99.9, type="label")
              for label_path in labels_path]
    assert len(images) == len(labels), "原图像数量和标注图像数量不一致！"

    # 初始化数据结构
    all_values = torch.IntTensor([])
    foreground_values = torch.IntTensor([])
    # 遍历所有图像
    for i in range(len(images)):
        # 获取当前图像的原图像和标注图像
        image = images[i]
        label = labels[i]
        # 累计所有灰度值
        all_values = torch.cat([all_values, image.flatten()], dim=0)
        # 累计前景灰度值
        foreground_values = torch.cat([foreground_values, image[label.nonzero(as_tuple=True)]], dim=0)

    # 分别获取所有灰度值的最大值和最小值
    all_min_val, all_max_val = all_values.min(), all_values.max()
    # 分别获取前景灰度值的最大值和最小值
    foreground_min_val, foreground_max_val = foreground_values.min(), foreground_values.max()
    # 归一化数值
    all_values = (all_values - all_min_val) / (all_max_val - all_min_val)
    foreground_values = (foreground_values - foreground_min_val) / (foreground_max_val - foreground_min_val)

    # 从小到大排序
    foreground_values = torch.sort(foreground_values)[0]
    # 转换格式
    foreground_values_np = foreground_values.numpy()
    # 获取两个阈值
    th_lower = foreground_values_np[int(0.004 * len(foreground_values_np))]
    th_upper = foreground_values_np[int(0.994 * len(foreground_values_np))]
    # # clip前景灰度值
    foreground_values_np[foreground_values_np < th_lower] = th_lower
    foreground_values_np[foreground_values_np > th_upper] = th_upper

    # 计算所有灰度值的均值和标准差
    all_mean, all_std = all_values.mean(), all_values.std()
    # 计算前景灰度值的均值和标准差
    foreground_mean, foreground_std = foreground_values_np.mean(), foreground_values_np.std()

    print("均值的取值范围为：[{}, {}]".format(all_mean, foreground_mean))
    print("标准差的取值范围为：[{}, {}]".format(foreground_std, all_std))



def calc_class_proportion(dataset_path=r"./datasets/src_10"):
    # 加载数据集中所有标注图像
    labels_path = glob.glob(os.path.join(dataset_path, "*", "labels", '*.nrrd'))
    # 对所有标注图像进行重采样，统一spacing
    labels = [utils.load_image_or_label(label_path, [0.5, 0.5, 0.5], 30, 99.9, type="label")
              for label_path in labels_path]

    # 加载类别名称和类别索引映射字典
    # 读取索引文件
    index_to_class_dict = utils.load_json_file(r"./3DTooth.json")
    class_to_index_dict = {}
    # 获得类别到索引的映射字典
    for key, val in index_to_class_dict.items():
        class_to_index_dict[val] = key

    # 初始化类别统计字典
    class_statistic_dict = {class_index: 0 for class_index in range(35)}

    # 遍历所有标注图像
    for label_i, label in enumerate(labels):
        # 获取当前标注图像的统计数据
        class_indexes, indexes_cnt = torch.unique(label, return_counts=True)
        # 遍历类别索引，将统计数据累加到类别统计字典
        for i, _ in enumerate(class_indexes):
            class_index = class_indexes[i].item()
            class_statistic_dict[class_index] += indexes_cnt[i].item()  # 类别体素数累加
    # 对统计字典按键排序
    class_statistic_dict = dict(sorted(class_statistic_dict.items(), key=lambda x: x[0]))

    # 初始化权重向量
    weights = np.zeros((35, ))
    # 依次计算每个类别的权重
    for ind, num in class_statistic_dict.items():
        if num != 0:
            weights[ind] = 1 / num
    print(weights)
    # 归一化权重数组
    weights = weights / weights.sum()
    print("各类别的权重数组为：", weights)









if __name__ == '__main__':

    # 显示所有图像文件的属性和值
    # show_space_property(r"./datasets/src_10")

    # 分析数据集每颗牙齿的统计数据和每张图像的统计数据
    # analyse_dataset(dataset_path=r"./datasets/src_10")

    # 计算clip的上下边界的范围,lower_bound:[all_min_val, foreground_min_val], upper_bound:[foreground_max_val, all_max_val]
    # calc_clip_bound()

    # 计算数据集的总均值、总标准差和前景均值、前景标准差
    # calc_mean_and_std()

    # 计算各类别的加权权重数组
    calc_class_proportion()















