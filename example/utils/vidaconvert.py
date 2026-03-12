import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
class ViDAInjectedLinear(nn.Module):
    def __init__(self,in_features, out_features, bias=False,):
        super().__init__()
        self.scale1=1.0
        self.scale2=1.0
        self.r=in_features//4
        self.r2=out_features*4
        self.linear_vida = nn.Linear(in_features, out_features, bias)
        self.vida_down = nn.Linear(in_features, self.r, bias=False)
        self.vida_up = nn.Linear(self.r, out_features, bias=False)
        self.vida_down2 = nn.Linear(in_features, self.r2, bias=False)
        self.vida_up2 = nn.Linear(self.r2, out_features, bias=False)
        nn.init.normal_(self.vida_down.weight, std=1 / self.r**2)
        nn.init.zeros_(self.vida_up.weight)

        nn.init.normal_(self.vida_down2.weight, std=1 / self.r2**2)
        nn.init.zeros_(self.vida_up2.weight)
    def forward(self,x):
        out=self.linear_vida(x)
        vida_out1=self.vida_up(self.vida_down(x))
        vida_out2=self.vida_up2(self.vida_down2(x))
        out=out+self.scale1*vida_out1+self.scale2*vida_out2
        return out

class ViDAInjectedConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.scale1=1.0
        self.scale2=1.0
        self.r=max(1, in_channels // 2)
        self.r2=out_channels*2
        self.conv_vida = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias, stride=stride, padding=padding)
        self.vida_down = nn.Conv2d(in_channels, self.r, kernel_size=1, bias=False, stride=1, padding=padding)
        self.vida_up = nn.Conv2d(self.r, out_channels, kernel_size=1, bias=False, stride=stride, padding=0)
        self.vida_down2 = nn.Conv2d(in_channels, self.r2, kernel_size=1, bias=False, stride=1, padding=padding)
        self.vida_up2 = nn.Conv2d(self.r2, out_channels, kernel_size=1, bias=False, stride=stride, padding=0)
        nn.init.normal_(self.vida_down.weight, std=1 / self.vida_down.weight.size(1)**2)
        nn.init.zeros_(self.vida_up.weight)
        nn.init.normal_(self.vida_down2.weight, std=1 / self.vida_down2.weight.size(1)**2)
        nn.init.zeros_(self.vida_up2.weight)
    
    def forward(self, x):
        original_out = self.conv_vida(x)
        
        vida_out1 = self.vida_up(self.vida_down(x)) * self.scale1
        vida_out2 = self.vida_up2(self.vida_down2(x)) * self.scale2
        
        out = original_out + vida_out1 + vida_out2
        
        return out

class ViDAInjectedConv1x1_Spatial(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.scale1 = 1.0
        self.scale2 = 1.0

        self.conv_vida = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
        self.spatial_down1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.spatial_up1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, groups=in_channels,bias=False)

        self.spatial_up2 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, groups=in_channels,bias=False)
        self.spatial_down2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        
        self.channel_adjust1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, stride=stride, padding=padding)
        self.channel_adjust2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, stride=stride, padding=padding)
    
    def forward(self, x):
        original_out = self.conv_vida(x)
        
        # 分支1：压缩-解压缩
        down1 = self.spatial_down1(x)
        up1 = self.spatial_up1(down1)
        branch1 = self.channel_adjust1(up1) * self.scale1
        
        # 分支2：扩展-压缩
        up2 = self.spatial_up2(x)
        down2 = self.spatial_down2(up2)
        branch2 = self.channel_adjust2(down2) * self.scale2
        
        # 确保尺寸匹配
        if branch1.size(2) != original_out.size(2) or branch1.size(3) != original_out.size(3):
            print("Upsampling branch1 from", branch1.size(), "to", original_out.size())
            branch1 = F.interpolate(branch1, size=original_out.shape[2:], mode='bilinear', align_corners=True)
        
        if branch2.size(2) != original_out.size(2) or branch2.size(3) != original_out.size(3):
            print("Upsampling branch2 from", branch2.size(), "to", original_out.size())
            branch2 = F.interpolate(branch2, size=original_out.shape[2:], mode='bilinear', align_corners=True)
        
        out = original_out + branch1 + branch2
        return out

def replace_1x1_conv_with_vida(model, exclude_layers=None):
    if exclude_layers is None:
        exclude_layers = []
    vida_params = []
    replaced_names = []
    model_device = next(model.parameters()).device
    def _replace_module(module, name_path=""):
        nonlocal vida_params, replaced_names
        
        for name, child in module.named_children():
            full_name = f"{name_path}.{name}" if name_path else name
            
            # 检查是否是 1x1 卷积且不在排除列表中
            if (isinstance(child, nn.Conv2d) and (child.kernel_size == (1, 1) or child.kernel_size == 1) and full_name not in exclude_layers):
                print(f"Replacing layer: {child.kernel_size},{child.stride} with ViDAInjectedConv1x1")
                # 保存原始参数
                in_channels = child.in_channels
                out_channels = child.out_channels
                has_bias = child.bias is not None
                stride = child.stride
                padding = child.padding
                dilation = child.dilation
                groups = child.groups
                # 创建 ViDA 注入的 1x1 卷积层
                vida_conv = ViDAInjectedConv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=has_bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                vida_conv = vida_conv.to(model_device)
                # 转移原始权重和偏置
                vida_conv.conv_vida.weight.data = child.weight.data
                if has_bias:
                    vida_conv.conv_vida.bias.data = child.bias.data
                setattr(module, name, vida_conv)
                child_vida_params = []
                child_vida_params.extend(list(vida_conv.vida_up.parameters()))
                child_vida_params.extend(list(vida_conv.vida_down.parameters()))
                child_vida_params.extend(list(vida_conv.vida_up2.parameters()))
                child_vida_params.extend(list(vida_conv.vida_down2.parameters()))
                # vida_params.extend(child_vida_params) 
                for param in child_vida_params:
                    param.requires_grad = True
                replaced_names.append(full_name)
            else:
                _replace_module(child, full_name)
    _replace_module(model)
    return vida_params, replaced_names, model

def replace_1x1_conv_with_spatial_vida(model, exclude_layers=None):
    if exclude_layers is None:
        exclude_layers = []
    vida_params = []
    replaced_names = []
    model_device = next(model.parameters()).device
    
    def _replace_module(module, name_path=""):
        nonlocal vida_params, replaced_names
        
        for name, child in module.named_children():
            full_name = f"{name_path}.{name}" if name_path else name
            
            # 检查是否是 1x1 卷积且不在排除列表中
            if (isinstance(child, nn.Conv2d) and 
                (child.kernel_size == (1, 1) or child.kernel_size == 1) and 
                full_name not in exclude_layers):
                
                print(f"Replacing layer: {full_name} with ViDAInjectedConv1x1_Spatial")
                
                # 保存原始参数
                in_channels = child.in_channels
                out_channels = child.out_channels
                has_bias = child.bias is not None
                stride = child.stride
                padding = child.padding
                dilation = child.dilation
                groups = child.groups
                
                # 创建空间ViDA注入的 1x1 卷积层
                vida_conv = ViDAInjectedConv1x1_Spatial(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=has_bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                vida_conv = vida_conv.to(model_device)
                
                # 转移原始权重和偏置到主卷积
                vida_conv.conv_vida.weight.data = child.weight.data
                if has_bias:
                    vida_conv.conv_vida.bias.data = child.bias.data
                
                setattr(module, name, vida_conv)
                
                # 收集VIDA分支参数
                child_vida_params = []
                child_vida_params.extend(list(vida_conv.spatial_down1.parameters()))
                child_vida_params.extend(list(vida_conv.spatial_up1.parameters()))
                child_vida_params.extend(list(vida_conv.spatial_up2.parameters()))
                child_vida_params.extend(list(vida_conv.spatial_down2.parameters()))
                child_vida_params.extend(list(vida_conv.channel_adjust1.parameters()))
                child_vida_params.extend(list(vida_conv.channel_adjust2.parameters()))
                
                # 设置VIDA分支参数为可训练
                for param in child_vida_params:
                    param.requires_grad = True
                
                vida_params.extend(child_vida_params)
                replaced_names.append(full_name)
            else:
                _replace_module(child, full_name)
    
    _replace_module(model)
    return vida_params, replaced_names, model