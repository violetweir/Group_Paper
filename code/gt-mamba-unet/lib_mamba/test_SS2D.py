import torch
import torch.nn as nn
import time
import math
from vmambanew import SS2D, GroupedSS2D
# ---------------------------------------------------------
# 假设这里已经导入了你源码中的 SS2D 和 GroupedSS2D
# 如果放在同一个文件下，请确保上方有 SS2D 和 GroupedSS2D 的定义
# from lib_mamba.vmambanew import SS2D
# from your_file import GroupedSS2D 
# ---------------------------------------------------------

class UltraLight_SS2D_Wrapper(nn.Module):
    """
    路线 A：UltraLight VM-UNet 官方的极致轻量化方案。
    在 SS2D 外部直接将输入张量按通道切分成 4 块，送入 4 个 d_model 只有 1/4 的微型 SS2D，
    最后 Concat 拼接，并通过归一化与 1x1 卷积/线性层融合。
    """
    def __init__(self, d_model=96, num_chunks=4, d_state=16, ssm_ratio=2.0, channel_first=True, k_group=4, forward_type="v05_noz", **kwargs):
        super().__init__()
        self.num_chunks = num_chunks
        self.chunk_dim = d_model // num_chunks
        self.channel_first = channel_first
        
        # 实例化 4 个迷你的 SS2D，外围映射层 (in_proj/out_proj) 也被等比例缩小到了 1/4！
        self.chunks = nn.ModuleList([
            SS2D(
                d_model=self.chunk_dim, 
                d_state=d_state, 
                ssm_ratio=ssm_ratio, 
                channel_first=channel_first,
                k_group=k_group,
                forward_type=forward_type,
                **kwargs
            ) for _ in range(num_chunks)
        ])
        
        # 独立的残差缩放因子 (ParameterList 保证能被 optimizer 追踪)
        self.skip_scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_chunks)])
        
        # 最后的特征融合层 (等价于论文 PVM Layer 的 norm + proj)
        if channel_first:
            # (B, C, H, W) 格式下，GroupNorm(1, C) 等价于全局 LayerNorm
            self.norm = nn.GroupNorm(1, d_model)
            self.proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        else:
            self.norm = nn.LayerNorm(d_model)
            self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # 切分的维度：channel_first 时是 dim=1 (C), 否则是 dim=-1
        chunk_dim_idx = 1 if self.channel_first else -1
        
        # 1. 物理切分通道
        xs = torch.chunk(x, self.num_chunks, dim=chunk_dim_idx)
        
        outs = []
        for i in range(self.num_chunks):
            # 2. 依次通过小 SS2D 并加上带缩放因子的残差
            # skip_scales[i] 是标量，PyTorch 会自动广播(Broadcast)
            out_i = self.chunks[i](xs[i]) + self.skip_scales[i] * xs[i]
            outs.append(out_i)
            
        # 3. 在通道维度拼接回来
        out = torch.cat(outs, dim=chunk_dim_idx)
        
        # 4. 全局归一化与通道特征融合
        out = self.norm(out)
        out = self.proj(out)
        
        return out


# =====================================================================
# 下方为专业的 Benchmark 脚本
# =====================================================================

def benchmark_ss2d_module(module_name, module, input_shape, device='cuda', warmup=10, num_runs=50):
    """
    测试指标: 参数量, 前向+反向平均耗时, 前向峰值显存, 反向峰值显存
    """
    print(f"\n{'='*55}")
    print(f"🚀 开始测试模块: {module_name}")
    print(f"{'='*55}")
    
    module = module.to(device)
    module.train() # 开启训练模式，保留激活值计算图
    
    # 1. 计算参数量
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"📦 可训练参数量: {params / 1e6:.4f} M")

    # 准备假数据
    x = torch.randn(input_shape, device=device, requires_grad=True)

    # 2. 预热 (Warmup)
    print("🔥 正在预热 (Warmup)...")
    for _ in range(warmup):
        module.zero_grad(set_to_none=True)
        out = module(x)
        out.sum().backward()
    torch.cuda.synchronize()

    # 3. 速度测试
    print(f"⏱️  正在进行速度测试 ({num_runs} 次循环)...")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        module.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
            
        start_events[i].record()
        out = module(x)
        loss = out.sum()
        loss.backward()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = sum(times) / num_runs
    print(f"⚡ 平均单次 Forward+Backward 耗时: {avg_time:.2f} ms")

    # 4. 显存测试
    print("💾 正在进行显存占用测试...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 测前向显存
    out = module(x)
    fwd_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    # 测反向显存
    torch.cuda.reset_peak_memory_stats(device)
    loss = out.sum()
    loss.backward()
    bwd_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f"📈 前向传播峰值显存 (Forward Peak VRAM):  {fwd_mem:.2f} MB")
    print(f"📉 反向传播峰值显存 (Backward Peak VRAM): {bwd_mem:.2f} MB")
    print(f"{'='*55}\n")
    
    return params, avg_time, fwd_mem, bwd_mem

if __name__ == "__main__":
    # --- 测试超参数配置 ---
    BATCH_SIZE = 8
    CHANNELS = 96      # vmamba_tiny 通道数
    HEIGHT = 64        # 假设 down3 / down4 的特征图尺寸
    WIDTH = 64
    INPUT_SHAPE = (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    
    FORWARD_TYPE = "v052d" # <--- 修正了之前报错的根源，强制指定合法的 forward_type

    # 1. 实例化原版 SS2D
    original_ss2d = SS2D(
        d_model=CHANNELS, 
        d_state=16, 
        ssm_ratio=2.0, 
        channel_first=True,  
        k_group=4,
        forward_type=FORWARD_TYPE
    )
    
    # 2. 实例化内卷版 GroupedSS2D (保留了外围特征融合，切分了内部状态机)
    # 注意：确保你的代码上下文中已经包含了昨天写的 GroupedSS2D
    try:
        grouped_ss2d = GroupedSS2D(
            num_chunks=4, 
            d_model=CHANNELS, 
            d_state=16, 
            ssm_ratio=2.0, 
            channel_first=True,
            k_group=4,
            forward_type=FORWARD_TYPE
        )
    except NameError:
        print("未检测到 GroupedSS2D 类，跳过该项测试。")
        grouped_ss2d = None

    # 3. 实例化外包版 UltraLight_SS2D_Wrapper (彻底的论文轻量化复刻)
    ultralight_ss2d = UltraLight_SS2D_Wrapper(
        num_chunks=4, 
        d_model=CHANNELS, 
        d_state=16, 
        ssm_ratio=2.0, 
        channel_first=True,
        k_group=4,
        forward_type=FORWARD_TYPE
    )

    # --- 开始大比武 ---
    print(f"正在测试输入张量尺寸: {INPUT_SHAPE}")
    
    try:
        benchmark_ss2d_module("1. 原版 SS2D (Original)", original_ss2d, INPUT_SHAPE)
        if grouped_ss2d:
            benchmark_ss2d_module("2. 内部切分多头 SS2D (GroupedSS2D)", grouped_ss2d, INPUT_SHAPE)
        benchmark_ss2d_module("3. 外部切分极致轻量版 (UltraLight_SS2D_Wrapper)", ultralight_ss2d, INPUT_SHAPE)
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")