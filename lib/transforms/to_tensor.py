# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/11/8 17:09
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch



class ToTensor(object):
    def __init__(self):
        pass



    def __call__(self, img_numpy, label):
        """
        Args:
            img_numpy: Image transforms from numpy to tensor
            label: Label segmentation map transforms from numpy to tensor

        Returns:
        """
        # 获取图像最大值和最小值
        max_val, min_val = img_numpy.max(), img_numpy.min()
        # 将图像灰度值归一化到0~1
        img_numpy = (img_numpy - min_val) / (max_val - min_val)
        return torch.FloatTensor(img_numpy), torch.FloatTensor(label)




