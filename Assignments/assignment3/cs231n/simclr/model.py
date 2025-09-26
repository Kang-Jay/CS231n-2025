import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    """SimCLR模型类，实现了自监督对比学习的网络结构"""
    
    def __init__(self, feature_dim=128):
        """
        初始化SimCLR模型
        
        Args:
            feature_dim (int): 投影头输出特征的维度，默认为128
        """
        super(Model, self).__init__()

        # 构建编码器backbone，基于ResNet50
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                # 修改第一个卷积层：将7x7卷积改为3x3，stride从2改为1
                # 这样做是为了避免在小图像上丢失太多信息
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # 排除所有的全连接层和最大池化层
            # 保留所有卷积层、批归一化层、ReLU激活层和自适应平均池化层
            # 这样可以保持特征提取能力，同时避免维度固定的限制
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # 编码器：提取图像特征的backbone网络
        self.f = nn.Sequential(*self.f)
        
        # 投影头：将编码器输出的特征映射到对比学习空间
        # 结构：2048 -> 512 -> feature_dim，使用BatchNorm和ReLU激活
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量
            
        Returns:
            tuple: (归一化的编码器特征, 归一化的投影头特征)
                - 编码器特征用于下游任务的线性评估
                - 投影头特征用于对比学习的损失计算
        """
        # 通过编码器提取特征
        x = self.f(x)
        # 展平特征图为向量
        feature = torch.flatten(x, start_dim=1)
        # 通过投影头得到对比学习特征
        out = self.g(feature)
        # 返回L2归一化后的特征，用于余弦相似度计算
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
