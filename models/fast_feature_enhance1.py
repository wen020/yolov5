import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv

class FeatureEnhanceModule(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c2 or c1
        # 三个特征提取分支
        self.edge_branch = EdgeBranch(c1, c2)
        self.texture_branch = TextureBranch(c1, c2)
        self.semantic_branch = SemanticBranch(c1, c2)
        # 自适应融合模块
        self.feature_aggregation = AdaptiveFusionModule(c2, c2)
        self.shortcut = Conv(c1, c2, k=1) if c1 != c2 else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        edge_feat = self.edge_branch(x)
        texture_feat = self.texture_branch(x)
        semantic_feat = self.semantic_branch(x)
        feats = torch.cat([edge_feat, texture_feat, semantic_feat], dim=1)
        out = self.feature_aggregation(feats)
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
        
    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return self.edge_conv(edge)

class TextureBranch(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # 多尺度卷积
        self.conv1 = Conv(c1, c2//2, k=3, d=1)
        self.conv2 = Conv(c1, c2//2, k=3, d=2)
        
    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        return torch.cat([feat1, feat2], dim=1)

class SemanticBranch(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = Conv(c1, c2, k=3)
        self.transformer = TransformerBlock(c2, 4)
        
    def forward(self, x):
        x = self.conv(x)
        return self.transformer(x)

class AdaptiveFusionModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 特征降维
        self.reduce_conv = Conv(c1 * 3, c1, k=1)
        
        # 特征重要性评估
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1//4, 1),
            nn.ReLU(),
            nn.Conv2d(c1//4, 3, 1),
            nn.Flatten(1),
            nn.Softmax(dim=1)
        )
        
        # 多尺度特征提取
        self.branch1 = nn.Sequential(
            Conv(c1, c2, k=3, d=1),
            TransformerBlock(c2, 4)
        )
        self.branch2 = Conv(c1, c2, k=3, d=2)
        self.branch3 = Conv(c1, c2, k=3, d=3)
        
    def forward(self, x):
        x = self.reduce_conv(x)
        weights = self.importance_net(x).view(-1, 3, 1, 1, 1)
        weights = weights / self.temperature
        
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        
        feats = torch.stack([feat1, feat2, feat3], dim=1)
        out = (feats * weights).sum(dim=1)
        return out

class TransformerBlock(nn.Module):
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
        
        # 确保通道数能被 head 数整除
        assert c % self.num_heads == 0, f'通道数 {c} 必须能被 head 数 {self.num_heads} 整除'
        
        # 重新排列维度
        x = x.view(b, self.num_heads, c // self.num_heads, h, w)
        # [b, heads, c//heads, h, w] -> [b, heads, h, w, c//heads]
        x = x.permute(0, 1, 3, 4, 2)
        
        # 计算注意力
        attn = F.softmax(x @ x.transpose(-2, -1) / math.sqrt(c // self.num_heads), dim=-1)
        x = attn @ x
        
        # 恢复维度
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(b, c, h, w)
        
        x = self.conv2(x)
        x = self.conv3(x)
        return x + shortcut