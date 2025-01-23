import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv

class FeatureEnhanceModule(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c2 or c1
        self.edge_branch = EdgeBranch(c1, c2)
        self.texture_branch = LightTextureBranch(c1, c2)
        self.semantic_branch = LightSemanticBranch(c1, c2)
        self.feature_aggregation = AdaptiveFusionModule(c2)
        self.shortcut = Conv(c1, c2, k=1) if c1 != c2 else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        edge_feat = self.edge_branch(x)
        texture_feat = self.texture_branch(x)
        semantic_feat = self.semantic_branch(x)
        out = self.feature_aggregation([edge_feat, texture_feat, semantic_feat])
        return out + identity

class EdgeBranch(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # Sobel算子
        self.sobel_x = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False)
        self.sobel_y = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False)
        
        with torch.no_grad():
            self.sobel_x.weight[:, :, :, :] = torch.tensor([[-1, 0, 1],
                                                          [-2, 0, 2],
                                                          [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.sobel_y.weight[:, :, :, :] = torch.tensor([[-1, -2, -1],
                                                          [0, 0, 0],
                                                          [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.edge_conv = Conv(c1, c2, k=3)
        self.se = SEBlock(c2)
        
    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        sobel_edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        learned_edge = self.edge_conv(x)
        edge_feat = sobel_edge + learned_edge
        return self.se(edge_feat)

class LightTextureBranch(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        mid_c = c2 // 4
        self.multi_scale_convs = nn.ModuleList([
            Conv(c1, mid_c, k=k, p=k//2) for k in [3, 5, 7]
        ])
        self.dcn = LightDeformConv(c2 * 3//4, c2)
        self.se = SEBlock(c2)
        
    def forward(self, x):
        feats = [conv(x) for conv in self.multi_scale_convs]
        x = torch.cat(feats, dim=1)
        x = self.dcn(x)
        return self.se(x)

class LightSemanticBranch(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.aspp = LightASPP(c1, c2)
        self.transformer = EfficientTransformerBlock(c2)
        self.se = SEBlock(c2)
        
    def forward(self, x):
        x = self.aspp(x)
        x = self.transformer(x)
        return self.se(x)

class AdaptiveFusionModule(nn.Module):
    """动态自适应特征融合模块"""
    def __init__(self, c1, c2):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))  # 温度参数
        
        # 特征降维
        self.reduce_conv = Conv(c1 * 3, c1, k=1)
        
        # 特征重要性评估网络
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1//4, 1),
            nn.ReLU(),
            nn.Conv2d(c1//4, 3, 1),
            nn.Flatten(1),
            nn.Softmax(dim=1)
        )
        
        # 多尺度特征提取分支 - 简化版本
        self.branch1 = nn.Sequential(
            Conv(c1, c2, 3),
            TransformerBlock(c2, 4)  # 保留transformer，体现注意力机制
        )
        
        self.branch2 = nn.Sequential(
            Conv(c1, c2, 3, d=2),  # 不同空洞率的卷积，体现多尺度
            nn.Conv2d(c2, c2, 1)
        )
        
        self.branch3 = nn.Sequential(
            Conv(c1, c2, 3, d=3),
            nn.Conv2d(c2, c2, 1)
        )
        
    def forward(self, x):
        x = self.reduce_conv(x)
        
        # 动态计算各分支权重
        weights = self.importance_net(x).view(-1, 3, 1, 1, 1)
        weights = weights / self.temperature
        
        # 多分支特征提取
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        
        # 堆叠特征并动态加权融合
        feats = torch.stack([feat1, feat2, feat3], dim=1)
        out = (feats * weights).sum(dim=1)
        return out

class TransformerBlock(nn.Module):
    """轻量级Transformer块"""
    def __init__(self, c, num_heads):
        super().__init__()
        self.conv1 = Conv(c, c, k=1)
        self.conv2 = Conv(c, c, k=3, p=1, g=c)
        self.conv3 = Conv(c, c, k=1)
        self.num_heads = num_heads
        
    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        b, c, h, w = x.shape
        x = x.view(b, self.num_heads, c // self.num_heads, h, w)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = F.softmax(x, dim=-1) @ x.transpose(-2, -1)
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(b, c, h, w)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + shortcut

class LightASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 8
        
        self.conv1 = Conv(in_channels, mid_channels, k=1)
        self.conv2 = Conv(in_channels, mid_channels, k=3, p=6, d=6, g=mid_channels)
        self.conv3 = Conv(in_channels, mid_channels, k=3, p=12, d=12, g=mid_channels)
        
        # 修改全局池化分支，使用简单的卷积而不是 Conv 类
        self.pool_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(inplace=True)
        )
        
        self.project = Conv(mid_channels * 4, out_channels, k=1)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        
        # 使用全局平均池化
        feat4 = x.mean([2, 3], keepdim=True)
        feat4 = self.pool_conv(feat4)
        feat4 = F.interpolate(feat4, size=size, mode='bilinear', align_corners=False)
        
        return self.project(torch.cat([feat1, feat2, feat3, feat4], dim=1))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 使用全局平均池化替代 AdaptiveAvgPool2d
        y = x.mean([2, 3])
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class EfficientTransformerBlock(nn.Module):
    def __init__(self, c, num_heads=8):
        super().__init__()
        self.conv1 = Conv(c, c, k=1)
        self.conv2 = Conv(c, c, k=3, p=1, g=c)
        self.conv3 = Conv(c, c, k=1)
        
    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + shortcut

class LightDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, k=3, p=1)
        self.conv2 = Conv(in_channels, 2, k=3, p=1)
        
    def forward(self, x):
        offset = self.conv2(x).sigmoid() * 2 - 1
        return self.conv1(x + offset.repeat(1, x.size(1)//2, 1, 1))