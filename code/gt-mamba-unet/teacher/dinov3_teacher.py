import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as TF
import logging

_logger = logging.getLogger(__name__)

class MKUNetDINOv3Teacher(nn.Module):
    def __init__(self,
                 dinov3_repo_path: str,          # 本地 DINOv3 代码仓库的路径
                 dinov3_weights_path: str,       # DINOv3 预训练权重路径
                 model_name: str = 'dinov3_vitb16',
                 patch_size: int = 16,
                 target_downsample_factor: int = 16): # 目标对齐的降采样倍率 (例如对应 Encoder4 的 1/16)
        super().__init__()
        self.patch_size = patch_size
        self.target_downsample_factor = target_downsample_factor

        _logger.info(f"[Teacher Model] Loading DINOv3 from local repo: {dinov3_repo_path}")
        
        try:
            # 必须使用 local 模式加载本地仓库的 DINOv3
            self.model = torch.hub.load(
                dinov3_repo_path,
                model_name,
                source='local',
                weights=dinov3_weights_path
            )
            
            self.teacher_feature_dim = self.model.embed_dim # 通常 ViT-B 是 768
            
            # 冻结所有权重，防止在蒸馏时被更新
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
                
            _logger.info(f"[Teacher Model] Successfully loaded DINOv3. Feature dim: {self.teacher_feature_dim}")
            
        except Exception as e:
            _logger.error(f"[Teacher Model] Failed to load DINOv3: {e}")
            raise

        # ImageNet 归一化标准
        self.normalize_transform = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    def forward(self, images):
        B, C, H_in, W_in = images.shape
        
        # 【适配医学图像】如果您的输入是单通道灰度图，这里会自动复制为3通道
        if C == 1:
            images = images.repeat(1, 3, 1, 1)

        normalized_images = self.normalize_transform(images)

        # ==========================================
        # 核心改进：动态插值对齐 (替换原版粗暴的 AvgPool)
        # ==========================================
        target_H_out = torch.tensor(H_in / self.target_downsample_factor)
        target_W_out = torch.tensor(W_in / self.target_downsample_factor)

        target_H_vit_in = torch.round(target_H_out * self.patch_size).int().item()
        target_W_vit_in = torch.round(target_W_out * self.patch_size).int().item()

        processed_images = TF.resize(
            normalized_images,
            [target_H_vit_in, target_W_vit_in],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True
        )

        with torch.no_grad():
            # 提取 DINOv3 特征
            output_dict = self.model(processed_images, is_training=True, masks=None)
            patch_tokens = output_dict["x_norm_patchtokens"]

            B, N_patches, C_teacher = patch_tokens.shape

            # ==========================================
            # 修复原版 BUG：使用实际长宽计算，支持非正方形输入
            # ==========================================
            H_patches_out = processed_images.shape[2] // self.patch_size
            W_patches_out = processed_images.shape[3] // self.patch_size

            if H_patches_out * W_patches_out != N_patches:
                raise ValueError(
                    f"[Teacher Model] 尺寸不匹配! "
                    f"预期 {H_patches_out * W_patches_out} tokens, 实际得到 {N_patches} tokens."
                )

            # 重塑为 2D 空间特征图 [B, 768, H/16, W/16]
            teacher_feature_map = patch_tokens.permute(0, 2, 1).reshape(
                B, C_teacher, H_patches_out, W_patches_out
            )

            return teacher_feature_map.detach()