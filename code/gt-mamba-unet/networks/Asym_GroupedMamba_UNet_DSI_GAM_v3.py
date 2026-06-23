import torch
from torch import nn
import torch.nn.functional as F
import math
from functools import partial

# --- [依赖库导入] ---
from timm.models.layers import trunc_normal_tf_, DropPath
from timm.models.helpers import named_apply

# 【核心导入】引入 MobileMamba 的 SS2D 模块
from lib_mamba.groupss2d import GroupedSS2D

# 尝试导入底层的 Triton 算子供 GroupedSS2D 使用
try:
    from lib_mamba.csm_triton import cross_scan_fn, cross_merge_fn
    from lib_mamba.csm_tritonk2 import cross_scan_fn_k2, cross_merge_fn_k2
    from lib_mamba.csms6s import selective_scan_fn
except ImportError:
    pass

__all__ = [
    'Asym_GroupedMamba_UNet_T', 'Asym_GroupedMamba_UNet_S',
    'Asym_GroupedMamba_UNet', 'Asym_GroupedMamba_UNet_M',
    'Asym_GroupedMamba_UNet_L', 'Asym_GroupedMamba_UNet_XL'
]


# ==========================================
# 0. 基础工具、结构重参数化 & 初始化函数
# ==========================================
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2):
    act = act.lower()
    if act == 'relu': return nn.ReLU(inplace)
    elif act == 'relu6': return nn.ReLU6(inplace)
    elif act == 'leakyrelu': return nn.LeakyReLU(neg_slope, inplace)
    elif act == 'gelu': return nn.GELU()
    elif act == 'hswish': return nn.Hardswish(inplace)
    else: raise NotImplementedError(f'activation layer [{act}] is not found')

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.reshape(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.reshape(batchsize, -1, height, width)
    return x

def gcd(a, b):
    while b: a, b = b, a % b
    return a

# ==========================================
# 1. 注意力机制模块 (CA, SA, GAG)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes if out_planes else in_planes
        ratio = min(ratio, self.in_planes)
        self.reduced_channels = max(1, self.in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(in_planes, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(x)

class GroupedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1, activation='relu'):
        super(GroupedAttentionGate, self).__init__()
        groups = 1 if kernel_size == 1 else groups
        self.W_g = Conv2d_BN(F_g, F_int, ks=kernel_size, stride=1, pad=kernel_size // 2, groups=groups)
        self.W_x = Conv2d_BN(F_l, F_int, ks=kernel_size, stride=1, pad=kernel_size // 2, groups=groups)
        self.psi = nn.Sequential(
            Conv2d_BN(F_int, 1, ks=1, stride=1, pad=0),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        psi = self.activation(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


# ==========================================
# 2. 浅层模块: MultiKernel CNN
# ==========================================
class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MultiKernelDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                Conv2d_BN(self.in_channels, self.in_channels, ks=kernel_size, stride=stride, pad=kernel_size // 2, groups=self.in_channels),
                act_layer(activation, inplace=True)
            ) for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs

class MultiKernelInvertedResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion_factor=2.0, dw_parallel=True, add=True, kernel_sizes=[1,3,5], activation='relu6', drop_path=0.0):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip_connection = True if self.stride == 1 else False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ex_c = int(self.in_c * expansion_factor)
        
        self.pconv1 = nn.Sequential(
            Conv2d_BN(self.in_c, self.ex_c, ks=1, stride=1, pad=0),
            act_layer(activation, inplace=True)
        )
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(self.ex_c, self.kernel_sizes, self.stride, activation, dw_parallel=dw_parallel)

        if self.add:
            self.combined_channels = self.ex_c * 1
        else:
            self.combined_channels = self.ex_c * self.n_scales
            
        self.pconv2 = Conv2d_BN(self.combined_channels, self.out_c, ks=1, stride=1, pad=0)
        
        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)
            
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        
        if self.add: 
            dout = sum(dwconv_outs)
        else: 
            dout = torch.cat(dwconv_outs, dim=1)
            
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out = self.pconv2(dout)
        
        if self.use_skip_connection:
            if self.in_c != self.out_c: 
                x = self.conv1x1(x)
            return x + self.drop_path(out)
        else: 
            return out
# ==========================================
# 3. 独立降采样与上采样模块 (接管核心通道变换)
# ==========================================
class Downsample_PConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.dw = Conv2d_BN(in_c, in_c, ks=2, stride=2, pad=0, groups=in_c)
        self.pw = Conv2d_BN(in_c, out_c, ks=1, stride=1, pad=0) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.pw(self.dw(x))

class Upsample_PConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pw = Conv2d_BN(in_c, out_c, ks=1, stride=1, pad=0) if in_c != out_c else nn.Identity()

    def forward(self, x):
        x = self.pw(x)
        return F.relu(F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=False))


# ==========================================
# 4. 纯净版 Grouped Mamba 与 FFN
# ==========================================
class ShuffleFFN(nn.Module):
    def __init__(self, ed: int, h: int, groups: int = 4):
        super().__init__()
        self.groups = groups if ed % groups == 0 and h % groups == 0 else 1
        self.pw1 = Conv2d_BN(ed, h, ks=1, stride=1, pad=0, groups=self.groups)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = Conv2d_BN(h, ed, ks=1, stride=1, pad=0, groups=self.groups, bn_weight_init=0)

    @staticmethod
    def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
        if groups == 1: return x
        B, C, H, W = x.shape
        channels_per_group = C // groups
        x = x.reshape(B, groups, channels_per_group, H, W).transpose(1, 2).contiguous()
        return x.reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(x)
        x = self.act(x)
        x = self.channel_shuffle(x, self.groups)
        x = self.pw2(x)
        return x

class GroupedMambaInvertedResidualBlock(nn.Module):
    """
    纯净版 GroupedSS2D！严格遵守 in_c == out_c。
    无任何 proj 层，通道伸缩全盘交由 Downsample/Upsample 处理。
    """
    def __init__(self, in_c, out_c, expansion_factor=2.0, num_chunks=4, drop_path=0.0, ssm_ratio=2.0, shufflegroups=1, activation='relu6'):
        super().__init__()
        if in_c != out_c:
            raise ValueError(f"GroupedMamba block expects strictly in_c == out_c. Got {in_c} and {out_c}.")

        self.in_c = in_c
        self.out_c = out_c
        
        self.drop_path0 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        h_dim = max(in_c, int(in_c * expansion_factor))
        
        self.dw0 = Conv2d_BN(in_c, in_c, 3, 1, 1, groups=in_c, bn_weight_init=0.)
        self.ffn0 = ShuffleFFN(ed=in_c, h=h_dim, groups=shufflegroups)

        self.op = GroupedSS2D(
    num_chunks=num_chunks,
    d_model=self.in_c,
    d_state=1,
    ssm_ratio=ssm_ratio,
    channel_first=True,

    # 完整四向 Cross2D
    k_group=4,
    forward_type="v05",

    # 明确启用 DG-GAT
    use_dg_gat=True,
    gat_attn_dim=16,
    gat_dropout=0.0,
    gat_graph_mode="factorized",

    initialize="v0",
)

        self.dw1 = Conv2d_BN(in_c, in_c, 3, 1, 1, groups=in_c, bn_weight_init=0.)
        self.ffn1 = ShuffleFFN(ed=in_c, h=h_dim, groups=shufflegroups)

    def forward(self, x):
        shortcut0 = x
        out = self.dw0(x)
        out = shortcut0 + self.drop_path0(self.ffn0(out))

        shortcut1 = out
        out = shortcut1 + self.drop_path1(self.op(out))

        shortcut2 = out
        out = self.dw1(out)
        out = shortcut2 + self.drop_path2(self.ffn1(out))
        return out

def grouped_mamba_bottleneck(in_c, out_c, n, num_chunks=4, ssm_ratio=2.0, expansion_factor=1.0, shufflegroups=1, dpr=None):
    if dpr is None: dpr = [0.0] * n
    convs = []
    for i in range(n):
        convs.append(GroupedMambaInvertedResidualBlock(
            in_c, out_c, expansion_factor=expansion_factor, num_chunks=num_chunks, ssm_ratio=ssm_ratio, drop_path=dpr[i], shufflegroups=shufflegroups,
        ))
    return nn.Sequential(*convs)

def mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=1.0, dw_parallel=True, add=True, kernel_sizes=[1,3,5], activation='relu6', dpr=None):
    if dpr is None: dpr = [0.0] * n
    convs = []
    xx = MultiKernelInvertedResidualBlock(in_c, out_c, s, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes, activation=activation, drop_path=dpr[0])
    convs.append(xx)
    if n > 1:
        for i in range(1, n):
            xx = MultiKernelInvertedResidualBlock(out_c, out_c, 1, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes, activation=activation, drop_path=dpr[i])
            convs.append(xx) 
    return nn.Sequential(*convs) 

# ==========================================
# 5. Asym_GroupedMamba_UNet + DINOv3 DSI 基类架构
# 同步 No-DINO 消融版结构调整，并保留 decoder1 蒸馏特征输出
# ==========================================
class Asym_GroupedMamba_UNet_Base(nn.Module):
    def __init__(
        self,
        num_classes=1,
        in_channels=3,
        channels=[16, 32, 64, 96, 160],
        depths=[1, 1, 1, 1, 1],
        expansion_factor=1.0,
        ssm_ratio=2.0,
        gag_kernel=3,
        drop_path_rate=0.1,
        num_chunks=4,
        shufflegroups=1,
        teacher_dim=768,
        use_distill_feature=True,
    ):
        super().__init__()
        self.use_distill_feature = bool(use_distill_feature)
        self.teacher_dim = int(teacher_dim)


        enc_depths = list(depths)
        dec_depths = [depths[4], depths[3], depths[2], depths[1], depths[0]]
        total_depth = sum(enc_depths) + sum(dec_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        idx = 0
        # ================== [Encoder] ==================
        # MK Block 允许内部使用 pconv 处理，Mamba block 则纯净
        self.encoder1 = mk_irb_bottleneck(in_channels, channels[0], enc_depths[0], 1, expansion_factor=expansion_factor, dpr=dpr[idx:idx+enc_depths[0]])
        self.down1 = Downsample_PConv(channels[0], channels[0])  # -> t1 (channels[0])
        idx += enc_depths[0]

        self.encoder2 = mk_irb_bottleneck(channels[0], channels[1], enc_depths[1], 1, expansion_factor=expansion_factor, dpr=dpr[idx:idx+enc_depths[1]])
        self.down2 = Downsample_PConv(channels[1], channels[1])  # -> t2 (channels[1])
        idx += enc_depths[1]

        self.encoder3 = mk_irb_bottleneck(channels[1], channels[2], enc_depths[2], 1, expansion_factor=expansion_factor, dpr=dpr[idx:idx+enc_depths[2]])
        self.down3 = Downsample_PConv(channels[2], channels[2])  # -> t3 (channels[2])
        idx += enc_depths[2]
        
        # Mamba block 强制 in==out。通道跃迁发生在 down4
        self.encoder4 = grouped_mamba_bottleneck(channels[2], channels[2], enc_depths[3], num_chunks=num_chunks, ssm_ratio=ssm_ratio, expansion_factor=expansion_factor, shufflegroups=shufflegroups, dpr=dpr[idx:idx+enc_depths[3]])
        self.down4 = Downsample_PConv(channels[2], channels[3])  # -> t4 (channels[3]! 通道在这里变深！)
        idx += enc_depths[3]

        self.encoder5 = grouped_mamba_bottleneck(channels[3], channels[3], enc_depths[4], num_chunks=num_chunks, ssm_ratio=ssm_ratio, expansion_factor=expansion_factor, shufflegroups=shufflegroups, dpr=dpr[idx:idx+enc_depths[4]])
        self.down5 = Downsample_PConv(channels[3], channels[4])  # 进 Decoder (channels[4])
        idx += enc_depths[4]

        # 注意力门控 (严格对齐每个跳跃连接的通道)
        self.AG1 = GroupedAttentionGate(F_g=channels[3], F_l=channels[3], F_int=max(1, channels[3] // 2), kernel_size=gag_kernel, groups=max(1, channels[3] // 2))
        self.AG2 = GroupedAttentionGate(F_g=channels[2], F_l=channels[2], F_int=max(1, channels[2] // 2), kernel_size=gag_kernel, groups=max(1, channels[2] // 2))
        self.AG3 = GroupedAttentionGate(F_g=channels[1], F_l=channels[1], F_int=max(1, channels[1] // 2), kernel_size=gag_kernel, groups=max(1, channels[1] // 2))
        self.AG4 = GroupedAttentionGate(F_g=channels[0], F_l=channels[0], F_int=max(1, channels[0] // 2), kernel_size=gag_kernel, groups=max(1, channels[0] // 2))

        # ================== [Decoder] ==================
        # 通道跃迁均依靠 Upsample_PConv
        self.decoder1 = grouped_mamba_bottleneck(channels[4], channels[4], dec_depths[0], num_chunks=num_chunks, ssm_ratio=ssm_ratio, expansion_factor=expansion_factor, shufflegroups=shufflegroups, dpr=dpr[idx:idx+dec_depths[0]])
        self.up1 = Upsample_PConv(channels[4], channels[3])  # 160 -> 96，放大并匹配 t4
        idx += dec_depths[0]

        self.decoder2 = grouped_mamba_bottleneck(channels[3], channels[3], dec_depths[1], num_chunks=num_chunks, ssm_ratio=ssm_ratio, expansion_factor=expansion_factor, shufflegroups=shufflegroups, dpr=dpr[idx:idx+dec_depths[1]])
        self.up2 = Upsample_PConv(channels[3], channels[2])  # 96 -> 64，匹配 t3
        idx += dec_depths[1]
        
        self.decoder3 = mk_irb_bottleneck(channels[2], channels[2], dec_depths[2], 1, expansion_factor=expansion_factor, dpr=dpr[idx:idx+dec_depths[2]])
        self.up3 = Upsample_PConv(channels[2], channels[1])  # 64 -> 32，匹配 t2
        idx += dec_depths[2]

        self.decoder4 = mk_irb_bottleneck(channels[1], channels[1], dec_depths[3], 1, expansion_factor=expansion_factor, dpr=dpr[idx:idx+dec_depths[3]])
        self.up4 = Upsample_PConv(channels[1], channels[0])  # 32 -> 16，匹配 t1
        idx += dec_depths[3]

        self.decoder5 = mk_irb_bottleneck(channels[0], channels[0], dec_depths[4], 1, expansion_factor=expansion_factor, dpr=dpr[idx:idx+dec_depths[4]])
        self.up5 = Upsample_PConv(channels[0], channels[0])  # 最终强制拉回 256x256
        idx += dec_depths[4]

        # CA 通道对齐 (基于当前层融合前的通道数)
        self.CA1 = ChannelAttention(channels[4], ratio=16) 
        self.CA2 = ChannelAttention(channels[3], ratio=16) 
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)
        self.SA = SpatialAttention()

        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        # DINOv3 深层语义蒸馏投影头。
        # 训练阶段在 decoder1 后、up1 前取 H/32 深层解码语义特征。
        # use_distill_feature=False 时用于 No-DINO 消融，保持推理接口不变。
        if self.use_distill_feature and self.teacher_dim != channels[4]:
            self.distill_projector = nn.Sequential(
                nn.Conv2d(channels[4], self.teacher_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.teacher_dim),
            )
        else:
            self.distill_projector = nn.Identity()

    def deploy(self):
        for name, module in self.named_modules():
            if hasattr(module, 'fuse') and callable(getattr(module, 'fuse')):
                path = name.split('.')
                parent = self
                for p in path[:-1]: parent = getattr(parent, p)
                setattr(parent, path[-1], module.fuse())
        print("[Deploy] 所有 Conv2d_BN 层已成功吸融，模型已准备好高速推理。")
        return self

    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)

        # ================== [还原你真正的天才前向流] ==================
        out = self.encoder1(x)    # 256x256
        out = self.down1(out)     # -> 128x128
        t1 = out                  
        
        out = self.encoder2(out)
        out = self.down2(out)     # -> 64x64
        t2 = out 
        
        out = self.encoder3(out)
        out = self.down3(out)     # -> 32x32
        t3 = out 
        
        out = self.encoder4(out)
        out = self.down4(out)     # -> 16x16
        t4 = out 

        out = self.encoder5(out)
        out = self.down5(out)     # -> 8x8 (这里算力最密集，但分辨率极小)

        # ================== [极速 Decode & Skip Connect] ==================
        out = self.CA1(out) * out
        out = self.SA(out) * out
        out = self.decoder1(out)    # 跑在 H/32 上

        # 训练阶段按需将 decoder1 特征映射到 DINOv3 教师维度。
        student_distill_feature = None
        if self.training and self.use_distill_feature:
            student_distill_feature = self.distill_projector(out)

        out = self.up1(out)         # 通道转换到 t4(96)，插值回 H/16
        t4 = self.AG1(g=out, x=t4)  
        out = torch.add(out, t4)

        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = self.decoder2(out)    # 跑在 16x16 上
        out = self.up2(out)         # 通道转换到 t3(64)，插值回 32x32
        t3 = self.AG2(g=out, x=t3)     
        out = torch.add(out, t3)

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = self.decoder3(out)    # 跑在 32x32 上
        out = self.up3(out)         # 变到 t2(32)，回 64x64
        t2 = self.AG3(g=out, x=t2)
        out = torch.add(out, t2)

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = self.decoder4(out)    # 跑在 64x64 上
        out = self.up4(out)         # 变到 t1(16)，回 128x128
        t1 = self.AG4(g=out, x=t1)
        out = torch.add(out, t1)

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = self.decoder5(out)    # 跑在 128x128 上
        # 无跳跃连接，暴力拔回 256x256 甩出结果！彻底避开 256 计算陷阱！
        out = self.up5(out)

        p4 = self.out4(out)

        # 与原网络保持推理接口兼容：eval 时仍只返回 [p4]。
        # 只有 use_distill_feature=True 的训练阶段才额外返回 DINOv3 对齐特征。
        if self.training and self.use_distill_feature:
            return [p4], student_distill_feature
        return [p4]


# ==========================================
# 6. 各参数规格变体封装
# ==========================================
class Asym_GroupedMamba_UNet_T(Asym_GroupedMamba_UNet_Base):
    def __init__(self, num_classes=1, in_channels=3, drop_path_rate=0.1, **kwargs):
        super().__init__(num_classes=num_classes, in_channels=in_channels, channels=[4, 8, 16, 32, 64], expansion_factor=1, drop_path_rate=drop_path_rate, **kwargs)

class Asym_GroupedMamba_UNet_S(Asym_GroupedMamba_UNet_Base):
    def __init__(self, num_classes=1, in_channels=3, drop_path_rate=0.1, **kwargs):
        super().__init__(num_classes=num_classes, in_channels=in_channels, channels=[8, 16, 32, 64, 96], expansion_factor=1, drop_path_rate=drop_path_rate, **kwargs)

class Asym_GroupedMamba_UNet(Asym_GroupedMamba_UNet_Base): 
    def __init__(self, num_classes=1, in_channels=3, drop_path_rate=0.1, **kwargs):
        # Base 默认应用你指定的: 16, 32, 64, 96, 160
        super().__init__(num_classes=num_classes, in_channels=in_channels, channels=[16, 32, 64, 96, 160], drop_path_rate=drop_path_rate, **kwargs)

class Asym_GroupedMamba_UNet_M(Asym_GroupedMamba_UNet_Base):
    def __init__(self, num_classes=1, in_channels=3, drop_path_rate=0.1, **kwargs):
        super().__init__(num_classes=num_classes, in_channels=in_channels, channels=[32, 64, 128, 192, 320], shufflegroups=4, drop_path_rate=drop_path_rate, **kwargs)

class Asym_GroupedMamba_UNet_L(Asym_GroupedMamba_UNet_Base):
    def __init__(self, num_classes=1, in_channels=3, drop_path_rate=0.1, **kwargs):
        super().__init__(num_classes=num_classes, in_channels=in_channels, channels=[64, 128, 256, 384, 512], shufflegroups=4, drop_path_rate=drop_path_rate, **kwargs)


class Asym_GroupedMamba_UNet_XL(Asym_GroupedMamba_UNet_Base):
    def __init__(self, num_classes=1, in_channels=3, drop_path_rate=0.1, **kwargs):
        super().__init__(num_classes=num_classes, in_channels=in_channels, channels=[96, 192, 384, 512, 768], drop_path_rate=drop_path_rate, **kwargs)



# ==========================================
# 测试入口与 FLOPs/Params 统计
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print(">>> [测试] 正在初始化 Asym_GroupedMamba_UNet_DSI_GAM 系列变体...")
    print("=" * 55)

    try:
        from thop import profile, clever_format
        has_thop = True
    except ImportError:
        print("提示: 未检测到 `thop` 库，跳过 FLOPs 计算。(可执行 `pip install thop` 安装)")
        has_thop = False

    models_to_test = {
        "Tiny": Asym_GroupedMamba_UNet_T(),
        "Small": Asym_GroupedMamba_UNet_S(),
        "Base": Asym_GroupedMamba_UNet(),
        "Medium": Asym_GroupedMamba_UNet_M(),
        "Large": Asym_GroupedMamba_UNet_L(),
        "XLarge": Asym_GroupedMamba_UNet_XL()

    }

    dummy_input = torch.randn(1, 3, 352, 352)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = dummy_input.to(device)

    for name, model in models_to_test.items():
        model = model.to(device)
        model.eval()

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if has_thop:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

                flops_str, params_str = clever_format([flops, params], "%.3f")
                print(f"[{name:9s}] 初始化成功 | Params: {params_str:>10} | FLOPs: {flops_str:>10}")
            except Exception as e:
                print(f"[{name:9s}] 初始化成功 | Params: {params / 1e6:7.3f} M | FLOPs: 计算失败 ({e})")
        else:
            print(f"[{name:9s}] 初始化成功 | Params: {params / 1e6:7.3f} M")

    print("\n" + "=" * 55)
    print(f">>> [测试] 运行 Base 版本前向传播 (未融合)... 输入: {list(dummy_input.shape)}")
    base_model = models_to_test["Base"]
    with torch.no_grad():
        out = base_model(dummy_input)
    print(f">>> [测试] 前向传播成功! 输出: {list(out[0].shape)}\n")