import torch
# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py


def expand_as_one_hot(input, C):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :return: 5D output image (NxCxDxHxW)
    """
    if input.dim() == 5:
        return input
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    # scatter to get the one-hot tensor
    return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def compute_per_channel_dice(input, target, epsilon=1e-6, mode="extension"):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
        input: 网络输出的经过Sigmoid或者Softmax归一化的预测概率图
        target: 真实标签图
        epsilon: 分母最小值，确保除法的稳定性
        mode: DSC的计算方式，"standard":标准计算方式；"extension":扩展计算方式

    Returns:

    """

    # 判断预测图和标注图的尺寸是否一致
    assert input.size() == target.size(), "计算dsc时input和target的尺寸不一致!"

    # 都压缩成二维tensor,(C, B * H * W * D)
    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # 计算DSC的分子
    intersect = (input * target).sum(-1)

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    if mode == "extension":
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
    elif mode == "standard":
        denominator = input.sum(-1) + target.sum(-1)

    # 返回最终计算得到的DSC,(C, )
    return 2 * (intersect / denominator.clamp(min=epsilon))


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, H, W, D) -> (C, N * H * W * D)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)
