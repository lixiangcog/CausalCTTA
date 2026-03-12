import numpy as np
import torch
import torch.nn as nn

class CausalTrimming:
    def __init__(self, device='cuda'):
        self.device = device
        self.mean_vector = None
        
    def compute_representation_statistics_spatial(self, representations):
        """
        完全向量化的PCA分析
        """
        n_samples, C, H, W = representations.shape
        
        # 重塑为 (n_samples, H*W, C)
        representations_reshaped = representations.permute(0, 2, 3, 1).reshape(n_samples, H*W, C)
        
        # 计算均值 - 向量化
        n_augmented = n_samples - 1
        original_reshaped = representations_reshaped[0:1]  # (1, H*W, C)
        augmented_reshaped = representations_reshaped[1:]  # (n_augmented, H*W, C)
        
        spatial_means = (1/(n_augmented + 1)) * original_reshaped + \
                      (n_augmented/(n_augmented + 1)) * torch.mean(augmented_reshaped, dim=0, keepdim=True)
        
        # 中心化
        spatial_centered = representations_reshaped - spatial_means  # (n_samples, H*W, C)
        
        # 批量计算协方差矩阵和SVD
        spatial_centered_batch = spatial_centered.permute(1, 0, 2)  # (H*W, n_samples, C)
        
        # 批量计算协方差矩阵
        cov_matrices = torch.bmm(spatial_centered_batch.transpose(1, 2), spatial_centered_batch)  # (H*W, C, C)
        
        # 批量SVD
        U, S, Vh = torch.linalg.svd(cov_matrices, full_matrices=False)
        
        # 特征值是奇异值的平方
        eigenvalues_batch = S ** 2  # (H*W, C)
        eigenvectors_batch = Vh.transpose(1, 2)  # (H*W, C, C)
        
        # 保存结果
        self.eigenvalues = eigenvalues_batch.reshape(H, W, C)
        self.eigenvectors = eigenvectors_batch.reshape(H, W, C, C)
        
        return self.eigenvectors, self.eigenvalues
    
    def trim_representation_spatial(self, z, variance_threshold=0.9):
        """
        去除主要主成分，保留次要特征
        """
        if self.eigenvectors is None or self.eigenvalues is None:
            raise ValueError("必须先调用 compute_representation_statistics_spatial 计算PCA组件")
        
        _, C, H, W = z.shape
        
        # 确保PCA组件与z在同一个设备上
        eigenvalues = self.eigenvalues.to(z.device)
        eigenvectors = self.eigenvectors.to(z.device)
        
        # 重塑输入和PCA组件
        z_flat = z[0].permute(1, 2, 0).reshape(H*W, C)  # (H*W, C)
        evals_flat = eigenvalues.reshape(H*W, C)  # (H*W, C)
        evecs_flat = eigenvectors.reshape(H*W, C, C)  # (H*W, C, C)
        
        # 计算累计解释方差
        total_variance = torch.sum(evals_flat, dim=1, keepdim=True)  # (H*W, 1)
        explained_variance_ratio = evals_flat / (total_variance + 1e-8)  # (H*W, C)
        cumulative_variance = torch.cumsum(explained_variance_ratio, dim=1)  # (H*W, C)
        
        # 确定要保留的主成分数量 - 保留解释方差较小的成分
        # 找到第一个超过阈值的索引
        threshold_mask = cumulative_variance >= variance_threshold
        used_components = torch.argmax(threshold_mask.int(), dim=1)  # (H*W,)
        used_components[~threshold_mask.any(dim=1)] = C - 1
        
        # 修正：保留后面的成分，去除前面的成分
        # 要保留的成分数量 = 总成分数 - (used_components + 1)
        n_components_to_keep = C - (used_components + 1)
        n_components_to_keep = torch.clamp(n_components_to_keep, min=0)
        
        # 计算要修剪的成分数量（去除前面的主要成分）
        m_keep_flat = used_components + 1  # 去除前 used_components+1 个主要成分
        
        # 创建修剪掩码 - 标记哪些成分需要被保留（后面的次要成分）
        component_indices = torch.arange(C, device=z.device).unsqueeze(0)  # (1, C)
        # 保留从索引 (used_components+1) 开始的成分
        keep_mask = component_indices >= (used_components + 1).unsqueeze(1)  # (H*W, C)
        
        # 计算所有成分的投影系数
        coefficients = torch.bmm(z_flat.unsqueeze(1), evecs_flat).squeeze(1)  # (H*W, C)
        
        # 将需要去除的成分的系数置零（保留次要成分）
        coefficients_trimmed = coefficients * keep_mask.float()
        
        # 使用修剪后的系数重建表示
        z_trimmed_flat = torch.bmm(coefficients_trimmed.unsqueeze(1), evecs_flat.transpose(1, 2)).squeeze(1)
        
        # 重塑回原始形状
        trimmed_z = z_trimmed_flat.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0)
        m_keep = m_keep_flat.reshape(H, W)
        
        # 输出统计信息
        avg_kept = torch.mean(n_components_to_keep.float())
        print(f"空间修剪统计 - 平均保留主成分数量: {avg_kept:.2f}/{C}")
        print(f"空间修剪统计 - 平均去除主成分数量: {torch.mean(m_keep.float()):.2f}")
        
        return trimmed_z