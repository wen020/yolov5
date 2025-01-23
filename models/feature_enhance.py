import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv
from torch.utils.checkpoint import checkpoint

class FeatureEnhanceModule(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c2 or c1
        
        # 边缘分支
        self.edge_branch = EdgeBranch(c1, c2)
        # 纹理分支
        self.texture_branch = TextureBranch(c1, c2)
        # 语义分支
        self.semantic_branch = SemanticBranch(c1, c2)
        # 特征聚合
        self.feature_aggregation = FeatureAggregation(c2)
        
        # 残差连接
        self.shortcut = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        # 使用梯度检查点
        edge_feat = checkpoint(self.edge_branch, x)
        texture_feat = checkpoint(self.texture_branch, x)
        semantic_feat = checkpoint(self.semantic_branch, x)
        
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
        
        # 边缘特征提取
        self.edge_conv = nn.Sequential(
            Conv(c1, c2, k=3),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        
        # SE注意力
        self.se = SEBlock(c2)
        
    def forward(self, x):
        # Sobel边缘
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        sobel_edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        
        # 学习型边缘
        learned_edge = self.edge_conv(x)
        
        # 特征融合
        edge_feat = sobel_edge + learned_edge
        
        # SE注意力增强
        enhanced_edge = self.se(edge_feat)
        return enhanced_edge

class TextureBranch(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # 多尺度卷积
        self.multi_scale_convs = nn.ModuleList([
            Conv(c1, c2//4, k=k, p=k//2) for k in [3, 5, 7, 9]
        ])
        
        # 可变形卷积
        self.dcn = DeformableConv2d(c2, c2)
        
        # 纹理注意力
        self.texture_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2//4, c2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 多尺度特征
        multi_feats = [conv(x) for conv in self.multi_scale_convs]
        texture_feat = torch.cat(multi_feats, dim=1)
        
        # DCN增强
        dcn_feat = self.dcn(texture_feat)
        
        # 注意力加权
        attention = self.texture_attention(dcn_feat)
        enhanced_texture = dcn_feat * attention
        
        return enhanced_texture

class SemanticBranch(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # ASPP模块
        self.aspp = ASPP(c1, c2)
        
        # Transformer模块
        self.transformer = TransformerBlock(c2)
        
        # 全局上下文编码
        self.context_encoding = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2//4, c2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # ASPP特征
        aspp_feat = self.aspp(x)
        
        # 只在特征图足够大时使用Transformer
        if x.size(-1) > 1 and x.size(-2) > 1:
            trans_feat = self.transformer(aspp_feat)
        else:
            trans_feat = aspp_feat
            
        # 上下文增强
        context = self.context_encoding(trans_feat)
        enhanced_semantic = trans_feat * context
        
        return enhanced_semantic

class FeatureAggregation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            batch_first=True
        )
        
        self.transform = nn.Sequential(
            Conv(channels*3, channels, k=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
    def forward(self, features):
        # 特征拼接和转换
        cat_feat = torch.cat(features, dim=1)
        transformed = self.transform(cat_feat)
        
        # 降低空间分辨率进行注意力计算
        b, c, h, w = transformed.shape
        h_new, w_new = h, w
        
        if h * w > 64*64:  # 如果特征图太大，进行下采样
            scale_factor = min(1.0, (64*64/(h*w))**0.5)  # 使用 ** 0.5 替代 sqrt
            h_new = int(h * scale_factor)
            w_new = int(w * scale_factor)
            transformed_small = F.interpolate(transformed, size=(h_new, w_new), 
                                           mode='bilinear', align_corners=False)
        else:
            transformed_small = transformed
            
        # 注意力计算
        feat_flat = transformed_small.view(b, c, -1).permute(0, 2, 1)
        attended, _ = self.attention(feat_flat, feat_flat, feat_flat)
        
        # 恢复原始分辨率
        out = attended.permute(0, 2, 1).view(b, c, h_new, w_new)
        if h_new != h or w_new != w:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            
        return out

# 辅助模块
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.size()[2:]
        
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        
        # 修改全局池化分支
        feat5 = self.pool(x)
        if size[0] > 1:  # 只在特征图足够大时进行上采样
            feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=False)
        else:
            feat5 = feat5.expand(-1, -1, size[0], size[1])
            
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        return self.project(out)

class TransformerBlock(nn.Module):
    def __init__(self, c, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(c, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(c, c * 2),
            nn.ReLU(),
            nn.Linear(c * 2, c)
        )
        self.ln1 = nn.LayerNorm(c)
        self.ln2 = nn.LayerNorm(c)
        
    def forward(self, x):
        b, c, h, w = x.shape
        h_new, w_new = h, w
        
        # 降低空间分辨率
        if h * w > 64*64:
            scale_factor = min(1.0, (64*64/(h*w))**0.5)  # 使用 ** 0.5 替代 sqrt
            h_new = int(h * scale_factor)
            w_new = int(w * scale_factor)
            x = F.interpolate(x, size=(h_new, w_new), mode='bilinear', align_corners=False)
        
        # 注意力计算
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        x_ln = self.ln1(x_flat)
        attended, _ = self.attention(x_ln, x_ln, x_ln)
        attended = x_flat + attended
        
        # FFN
        out = attended + self.ffn(self.ln2(attended))
        
        # 重塑并恢复分辨率
        out = out.permute(0, 2, 1).view(b, c, h_new, w_new)
        if h_new != h or w_new != w:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        
        return out

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 修改 offset 卷积的输出通道数
        self.offset_conv = nn.Conv2d(in_channels, 2 * 3 * 3, 3, padding=1)  # 2*3*3 for x,y offsets
        self.mask_conv = nn.Conv2d(in_channels, 3 * 3, 3, padding=1)  # 3*3 for attention masks
        self.conv = Conv(in_channels, out_channels, k=3, p=1)
        
        # 初始化
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)
        
    def forward(self, x):
        # 计算 offset 和 mask
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        
        # 应用可变形卷积
        # 由于PyTorch原生不支持可变形卷积，这里用普通卷积模拟
        out = self.conv(x)
        b, c, h, w = out.shape
        
        # 使用grid_sample模拟可变形效果
        grid = self._get_grid(h, w, x.device)
        offset = offset.view(b, 2, 9, h, w)
        grid = grid.unsqueeze(0) + offset.mean(dim=2)
        
        # 归一化网格坐标
        grid = grid.permute(0, 2, 3, 1)
        out = F.grid_sample(x, grid, align_corners=True)
        
        return out
    
    def _get_grid(self, h, w, device):
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grid = torch.stack([xx, yy], dim=0).float()
        grid = 2.0 * grid / torch.tensor([w-1, h-1]).view(-1, 1, 1) - 1.0
        return grid.to(device)