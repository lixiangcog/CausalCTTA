import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))

class Projector(nn.Module):
    def __init__(self, word_dim=512, in_dim=32, out_dim=2, kernel_size=3):
        super().__init__()
        self.in_dim = 4
        self.out_dim = out_dim  
        self.kernel_size = kernel_size
        # 考虑到图片分辨率已经足够，不需要上采样扩大
        self.vis = nn.Sequential(
            conv_layer(in_dim, int(math.sqrt(in_dim * self.in_dim)), 3, padding=1),
            nn.Conv2d(int(math.sqrt(in_dim * self.in_dim)), self.in_dim, 1)
        )
        
        # 文本投影器保持不变
        text_dim = self.out_dim * self.in_dim * kernel_size * kernel_size + self.out_dim
        feat_dim = int(math.sqrt(word_dim * text_dim))
        feat_dim = 64
        self.txt = nn.Sequential(
            nn.Linear(word_dim, feat_dim),
            nn.LayerNorm(feat_dim),  # 稳定特征
            nn.GELU(),  # 更平滑的激活函数
            nn.Linear(feat_dim, text_dim)
        )

    def forward(self, x, word):
        """
        x: (1, 32, 512, 512) - 视觉特征
        word: (1, 512) - 文本特征
        """
        x = self.vis(x)  # (1, 32, 512, 512)
        params = self.txt(word)  # (1, out_dim*in_dim*k*k + out_dim)
        weight_params = params[:, :-self.out_dim]  # (1, out_dim*in_dim*k*k)
        bias_params = params[:, -self.out_dim:]    # (1, out_dim)
        
        weight = weight_params.reshape(self.out_dim,self.in_dim,self.kernel_size,self.kernel_size)  # (out_dim, in_dim, k, k)
        
        out = F.conv2d(x, weight, bias=bias_params.squeeze(0), padding=self.kernel_size // 2)
        
        return out
