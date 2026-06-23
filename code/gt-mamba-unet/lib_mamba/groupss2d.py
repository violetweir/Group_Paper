import math
import torch
from torch import nn
import torch.nn.functional as F

# 你的工程中应已经存在以下导入：
from .vmambanew import SS2D
from .csm_triton import cross_scan_fn, cross_merge_fn
from .csm_tritonk2 import cross_scan_fn_k2, cross_merge_fn_k2
from .csms6s import selective_scan_fn


class DirectionGroupGAT(nn.Module):
    """
    Direction-Group Graph Attention，简称 DG-GAT。

    将 K 个扫描方向 × M 个通道组视为 K*M 个图节点。

    输入：
        x: [B, K, M, Cg, H, W]

    输出：
        x: [B, K, M, Cg, H, W]

    graph_mode="factorized" 时：
        1. 同一扫描方向内的不同通道组相连；
        2. 同一通道组在不同扫描方向间相连；
        3. 每个节点包含自连接。

    graph_mode="full" 时：
        所有方向-通道组节点两两相连。
    """

    def __init__(
        self,
        num_directions: int,
        num_groups: int,
        node_dim: int,
        attn_dim: int = 16,
        dropout: float = 0.0,
        graph_mode: str = "factorized",
    ):
        super().__init__()

        if num_directions <= 0:
            raise ValueError(
                f"num_directions must be positive, got {num_directions}"
            )

        if num_groups <= 0:
            raise ValueError(
                f"num_groups must be positive, got {num_groups}"
            )

        if node_dim <= 0:
            raise ValueError(
                f"node_dim must be positive, got {node_dim}"
            )

        if graph_mode not in ("factorized", "full"):
            raise ValueError(
                f"Unsupported graph_mode: {graph_mode}"
            )

        self.num_directions = num_directions
        self.num_groups = num_groups
        self.num_nodes = num_directions * num_groups
        self.node_dim = node_dim
        self.graph_mode = graph_mode

        hidden_dim = max(1, min(attn_dim, node_dim))

        # 节点描述归一化与低维映射
        self.node_norm = nn.LayerNorm(node_dim)

        self.node_proj = nn.Linear(
            node_dim,
            hidden_dim,
            bias=False,
        )

        # 加性 GAT 注意力：
        # e_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j)
        self.attn_src = nn.Linear(
            hidden_dim,
            1,
            bias=False,
        )

        self.attn_dst = nn.Linear(
            hidden_dim,
            1,
            bias=False,
        )

        self.leaky_relu = nn.LeakyReLU(
            negative_slope=0.2
        )

        self.attn_dropout = nn.Dropout(dropout)

        # 对完整节点特征图进行共享 Value 投影。
        # 这样不同通道组的信息在聚合前可以先完成语义映射。
        self.value_proj = nn.Conv2d(
            node_dim,
            node_dim,
            kernel_size=1,
            bias=False,
        )

        # 初始化为0：
        # 初始状态下 DG-GAT 不改变原始 GroupedSS2D 输出。
        self.gamma = nn.Parameter(
            torch.zeros(1)
        )

        adjacency = self._build_adjacency()

        self.register_buffer(
            "adjacency",
            adjacency,
            persistent=False,
        )

    def _build_adjacency(self) -> torch.Tensor:
        """
        构建图邻接矩阵。

        factorized：
            同方向相连，或者同通道组相连。

        full：
            所有节点完全连接。
        """
        n = self.num_nodes

        if self.graph_mode == "full":
            return torch.ones(
                (n, n),
                dtype=torch.bool,
            )

        node_ids = torch.arange(n)

        # 节点排列顺序：
        # direction 0: group 0, 1, ..., M-1
        # direction 1: group 0, 1, ..., M-1
        direction_ids = (
            node_ids // self.num_groups
        )

        group_ids = (
            node_ids % self.num_groups
        )

        same_direction = (
            direction_ids[:, None]
            == direction_ids[None, :]
        )

        same_group = (
            group_ids[:, None]
            == group_ids[None, :]
        )

        # 两个关系都天然包含自连接
        adjacency = same_direction | same_group

        return adjacency

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, K, M, Cg, H, W]

        Returns:
            out: [B, K, M, Cg, H, W]
        """
        if x.ndim != 6:
            raise ValueError(
                "DirectionGroupGAT expects "
                f"[B,K,M,C,H,W], got {tuple(x.shape)}"
            )

        B, K, M, C, H, W = x.shape

        if K != self.num_directions:
            raise ValueError(
                f"Expected K={self.num_directions}, "
                f"got K={K}"
            )

        if M != self.num_groups:
            raise ValueError(
                f"Expected M={self.num_groups}, "
                f"got M={M}"
            )

        if C != self.node_dim:
            raise ValueError(
                f"Expected node_dim={self.node_dim}, "
                f"got C={C}"
            )

        N = K * M

        # [B,K,M,C,H,W] -> [B,N,C,H,W]
        nodes = x.reshape(
            B,
            N,
            C,
            H,
            W,
        )

        # 每个方向-通道组通过全局平均池化得到图节点描述
        descriptors = nodes.mean(
            dim=(-1, -2)
        )
        # descriptors: [B,N,C]

        h = self.node_proj(
            self.node_norm(descriptors)
        )
        # h: [B,N,attn_dim]

        src_score = self.attn_src(h)
        # [B,N,1]

        dst_score = self.attn_dst(h).transpose(
            1,
            2,
        )
        # [B,1,N]

        attention_logits = self.leaky_relu(
            src_score + dst_score
        )
        # [B,N,N]

        graph_mask = self.adjacency.unsqueeze(0)

        attention_logits = (
            attention_logits.masked_fill(
                ~graph_mask,
                torch.finfo(
                    attention_logits.dtype
                ).min,
            )
        )

        attention = F.softmax(
            attention_logits,
            dim=-1,
        )

        attention = self.attn_dropout(
            attention
        )

        # 对每个完整节点特征图进行共享Value变换
        values = self.value_proj(
            nodes.reshape(
                B * N,
                C,
                H,
                W,
            )
        )

        values = values.reshape(
            B,
            N,
            C,
            H,
            W,
        )

        # 根据图注意力权重聚合其他节点的完整特征图
        messages = torch.einsum(
            "bij,bjchw->bichw",
            attention,
            values,
        )

        # 残差式图信息传播
        # gamma初始为0，因此初始输出等于原始nodes
        nodes = nodes + self.gamma * messages

        return nodes.reshape(
            B,
            K,
            M,
            C,
            H,
            W,
        )


class GroupedSS2D(SS2D):
    """
    带 Direction-Group GAT 的 GroupedSS2D。

    主要流程：

        Input
          ↓
        Cross2D Scan
          ↓
        K个方向 × M个通道组
          ↓
        K*M组 Selective Scan
          ↓
        四方向空间对齐
          ↓
        Direction-Group GAT
          ↓
        四方向融合
          ↓
        Output Norm

    注意：
        DG-GAT 当前只在以下条件下启用：

            k_group = 4
            scan_mode = "cross2d"

        即你当前使用的：

            forward_type = "v05"
            k_group = 4

        其他扫描模式自动退化为原始 CrossMerge 流程。
    """

    def __init__(
        self,
        *args,
        num_chunks: int = 4,
        use_dg_gat: bool = True,
        gat_attn_dim: int = 16,
        gat_dropout: float = 0.0,
        gat_graph_mode: str = "factorized",
        **kwargs,
    ):
        if num_chunks <= 0:
            raise ValueError(
                f"num_chunks must be positive, "
                f"got {num_chunks}"
            )

        self.num_chunks = num_chunks
        self.use_dg_gat = use_dg_gat
        self.gat_attn_dim = gat_attn_dim
        self.gat_dropout = gat_dropout
        self.gat_graph_mode = gat_graph_mode

        super().__init__(*args, **kwargs)

    def __initv2__(
        self,
        **kwargs,
    ):
        """
        先调用父类初始化完整SS2D，再重建分组投影参数和DG-GAT。
        """
        super().__initv2__(**kwargs)

        d_model = kwargs.get(
            "d_model",
            96,
        )

        ssm_ratio = kwargs.get(
            "ssm_ratio",
            2.0,
        )

        d_state = kwargs.get(
            "d_state",
            16,
        )

        dt_rank = kwargs.get(
            "dt_rank",
            "auto",
        )

        k_group = kwargs.get(
            "k_group",
            4,
        )

        dt_min = kwargs.get(
            "dt_min",
            0.001,
        )

        dt_max = kwargs.get(
            "dt_max",
            0.1,
        )

        dt_init = kwargs.get(
            "dt_init",
            "random",
        )

        dt_scale = kwargs.get(
            "dt_scale",
            1.0,
        )

        dt_init_floor = kwargs.get(
            "dt_init_floor",
            1e-4,
        )

        initialize = kwargs.get(
            "initialize",
            "v0",
        )

        if k_group not in (2, 4):
            raise ValueError(
                f"k_group must be 2 or 4, "
                f"got {k_group}"
            )

        d_inner = int(
            ssm_ratio * d_model
        )

        if d_inner % self.num_chunks != 0:
            raise ValueError(
                f"d_inner({d_inner}) must be "
                f"divisible by "
                f"num_chunks({self.num_chunks})"
            )

        self.chunk_d_inner = (
            d_inner // self.num_chunks
        )

        if dt_rank == "auto":
            self.chunk_dt_rank = max(
                1,
                math.ceil(
                    (
                        d_model
                        / self.num_chunks
                    )
                    / 16
                ),
            )
        else:
            self.chunk_dt_rank = max(
                1,
                int(dt_rank)
                // self.num_chunks,
            )

        total_groups = (
            k_group * self.num_chunks
        )

        out_per_group = (
            self.chunk_dt_rank
            + 2 * d_state
        )

        # =====================================
        # 1. 分组 x_proj
        # =====================================
        #
        # 每个方向-通道组独立生成：
        #     low-rank dt
        #     B
        #     C
        #
        self.x_proj_weight = nn.Parameter(
            torch.empty(
                total_groups
                * out_per_group,
                self.chunk_d_inner,
                1,
            )
        )

        # 接近 nn.Linear 的默认初始化
        nn.init.kaiming_uniform_(
            self.x_proj_weight.squeeze(-1),
            a=math.sqrt(5),
        )

        # =====================================
        # 2. 分组 dt_proj
        # =====================================
        self.dt_projs_weight = nn.Parameter(
            torch.empty(
                total_groups
                * self.chunk_d_inner,
                self.chunk_dt_rank,
                1,
            )
        )

        self.dt_projs_bias = nn.Parameter(
            torch.empty(
                total_groups
                * self.chunk_d_inner
            )
        )

        self._init_grouped_dt(
            initialize=initialize,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
        )

        # =====================================
        # 3. Direction-Group GAT
        # =====================================
        if self.use_dg_gat:
            if k_group != 4:
                raise ValueError(
                    "DG-GAT currently requires "
                    "k_group=4 because it aligns "
                    "the four Cross2D directions."
                )

            self.dg_gat = DirectionGroupGAT(
                num_directions=k_group,
                num_groups=self.num_chunks,
                node_dim=self.chunk_d_inner,
                attn_dim=self.gat_attn_dim,
                dropout=self.gat_dropout,
                graph_mode=self.gat_graph_mode,
            )
        else:
            self.dg_gat = None

    def _init_grouped_dt(
        self,
        initialize: str,
        dt_init: str,
        dt_scale: float,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
    ) -> None:
        """
        为重新构造后的分组 dt 参数执行初始化。

        initialize="v0"：
            使用原始Mamba的dt初始化方式。

        initialize="v1"：
            使用0.1倍高斯随机初始化。

        initialize="v2"：
            使用[0,0.1)均匀随机初始化。
        """
        if initialize == "v0":
            dt_init_std = (
                self.chunk_dt_rank
                ** -0.5
                * dt_scale
            )

            if dt_init == "constant":
                nn.init.constant_(
                    self.dt_projs_weight,
                    dt_init_std,
                )

            elif dt_init == "random":
                nn.init.uniform_(
                    self.dt_projs_weight,
                    -dt_init_std,
                    dt_init_std,
                )

            else:
                raise ValueError(
                    f"Unsupported dt_init: "
                    f"{dt_init}"
                )

            # 让 softplus(dt_bias)
            # 初始化在 [dt_min, dt_max] 范围内
            dt = torch.exp(
                torch.rand_like(
                    self.dt_projs_bias
                )
                * (
                    math.log(dt_max)
                    - math.log(dt_min)
                )
                + math.log(dt_min)
            ).clamp(
                min=dt_init_floor
            )

            # inverse softplus
            inv_dt = (
                dt
                + torch.log(
                    -torch.expm1(-dt)
                )
            )

            with torch.no_grad():
                self.dt_projs_bias.copy_(
                    inv_dt
                )

        elif initialize == "v1":
            nn.init.normal_(
                self.dt_projs_weight,
                mean=0.0,
                std=0.1,
            )

            nn.init.normal_(
                self.dt_projs_bias,
                mean=0.0,
                std=0.1,
            )

        elif initialize == "v2":
            nn.init.uniform_(
                self.dt_projs_weight,
                0.0,
                0.1,
            )

            nn.init.uniform_(
                self.dt_projs_bias,
                0.0,
                0.1,
            )

        else:
            raise ValueError(
                f"Unsupported initialize mode: "
                f"{initialize}"
            )

    @staticmethod
    def _align_cross2d_outputs(
        ys: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        将 Cross2D 四个方向的扫描输出恢复到统一原图坐标。

        输入：
            ys: [B, 4, M, Cg, L]

        输出：
            aligned: [B, 4, M, Cg, H, W]

        四个方向对应：
            0：行优先正向
            1：列优先正向
            2：行优先反向
            3：列优先反向
        """
        if ys.ndim != 5:
            raise ValueError(
                "Expected ys=[B,4,M,C,L], "
                f"got {tuple(ys.shape)}"
            )

        B, K, M, C, L = ys.shape

        if K != 4:
            raise ValueError(
                "Cross2D alignment requires "
                f"K=4, got K={K}"
            )

        if L != H * W:
            raise ValueError(
                f"Expected L={H * W}, "
                f"got L={L}"
            )

        # 方向0：行优先正向
        y0 = ys[:, 0].reshape(
            B,
            M,
            C,
            H,
            W,
        )

        # 方向1：列优先正向
        # 扫描时相当于对转置后的 [W,H] 特征展开
        y1 = ys[:, 1].reshape(
            B,
            M,
            C,
            W,
            H,
        )

        y1 = y1.transpose(
            -2,
            -1,
        ).contiguous()

        # 方向2：行优先反向
        y2 = ys[:, 2].flip(
            dims=[-1]
        )

        y2 = y2.reshape(
            B,
            M,
            C,
            H,
            W,
        )

        # 方向3：列优先反向
        y3 = ys[:, 3].flip(
            dims=[-1]
        )

        y3 = y3.reshape(
            B,
            M,
            C,
            W,
            H,
        )

        y3 = y3.transpose(
            -2,
            -1,
        ).contiguous()

        return torch.stack(
            (y0, y1, y2, y3),
            dim=1,
        )

    def forward_corev2(
        self,
        x: torch.Tensor = None,
        force_fp32: bool = False,
        ssoflex: bool = True,
        no_einsum: bool = True,
        selective_scan_backend=None,
        scan_mode: str = "cross2d",
        scan_force_torch: bool = False,
        **kwargs,
    ):
        if x is None:
            raise ValueError(
                "x must not be None"
            )

        if scan_mode not in (
            "unidi",
            "bidi",
            "cross2d",
            "cascade2d",
        ):
            raise ValueError(
                f"Unsupported scan_mode: "
                f"{scan_mode}"
            )

        delta_softplus = True

        B, D, H, W = x.shape
        L = H * W

        K = self.k_group
        M = self.num_chunks

        cD = self.chunk_d_inner
        cR = self.chunk_dt_rank

        N = self.A_logs.shape[1]

        if K not in (2, 4):
            raise ValueError(
                f"k_group must be 2 or 4, "
                f"got {K}"
            )

        if D != M * cD:
            raise ValueError(
                f"Channel mismatch: D={D}, "
                f"but num_chunks*chunk_d_inner="
                f"{M}*{cD}={M * cD}"
            )

        if self.A_logs.shape[0] != K * D:
            raise ValueError(
                "A_logs first dimension must be "
                f"K*D={K * D}, got "
                f"{self.A_logs.shape[0]}"
            )

        if self.Ds.numel() != K * D:
            raise ValueError(
                f"Ds size must be K*D={K * D}, "
                f"got {self.Ds.numel()}"
            )

        scan_id = {
            "cross2d": 0,
            "unidi": 1,
            "bidi": 2,
            "cascade2d": 3,
        }[scan_mode]

        # =====================================
        # 1. 四向或双向空间扫描
        # =====================================
        if K == 4:
            xs = cross_scan_fn(
                x,
                in_channel_first=True,
                out_channel_first=True,
                scans=scan_id,
                force_torch=scan_force_torch,
            )
        else:
            xs = cross_scan_fn_k2(
                x,
                in_channel_first=True,
                out_channel_first=True,
                scans=scan_id,
                force_torch=scan_force_torch,
            )

        # xs:
        # [B,K,D,L]
        #
        # 划分为：
        # [B,K*M,cD,L]
        xs_grouped = xs.reshape(
            B,
            K * M,
            cD,
            L,
        )

        # =====================================
        # 2. 每个方向-通道组独立生成dt/B/C
        # =====================================
        x_dbl = F.conv1d(
            xs_grouped.reshape(
                B,
                K * M * cD,
                L,
            ),
            self.x_proj_weight,
            bias=None,
            groups=K * M,
        )

        x_dbl = x_dbl.reshape(
            B,
            K * M,
            cR + 2 * N,
            L,
        )

        dts, Bs, Cs = torch.split(
            x_dbl,
            [cR, N, N],
            dim=2,
        )

        # 低秩dt：
        # [B,K*M,cR,L]
        #
        # 映射为通道级dt：
        # [B,K*M*cD,L]
        dts = F.conv1d(
            dts.reshape(
                B,
                K * M * cR,
                L,
            ),
            self.dt_projs_weight,
            bias=None,
            groups=K * M,
        )

        u = xs_grouped.reshape(
            B,
            K * D,
            L,
        )

        delta = dts.reshape(
            B,
            K * D,
            L,
        )

        As = -self.A_logs.float().exp()
        Ds = self.Ds.float()

        Bs = Bs.reshape(
            B,
            K * M,
            N,
            L,
        )

        Cs = Cs.reshape(
            B,
            K * M,
            N,
            L,
        )

        delta_bias = (
            self.dt_projs_bias
            .reshape(-1)
            .float()
        )

        if force_fp32:
            u = u.float()
            delta = delta.float()
            Bs = Bs.float()
            Cs = Cs.float()

        def selective_scan(
            u_: torch.Tensor,
            delta_: torch.Tensor,
            A_: torch.Tensor,
            B_: torch.Tensor,
            C_: torch.Tensor,
            D_: torch.Tensor = None,
            delta_bias_: torch.Tensor = None,
        ):
            if u_.device.type == "cpu":
                backend = "torch"

            elif selective_scan_backend is not None:
                backend = selective_scan_backend

            else:
                backend = (
                    "oflex"
                    if ssoflex
                    else "mamba"
                )

            return selective_scan_fn(
                u_,
                delta_,
                A_,
                B_,
                C_,
                D_,
                delta_bias_,
                delta_softplus,
                ssoflex,
                backend=backend,
            )

        # =====================================
        # 3. Selective Scan
        # =====================================
        ys = selective_scan(
            u,
            delta,
            As,
            Bs,
            Cs,
            Ds,
            delta_bias,
        )

        # =====================================
        # 4. DG-GAT分支
        # =====================================
        if (
            self.use_dg_gat
            and K == 4
            and scan_mode == "cross2d"
        ):
            # [B,K*D,L]
            # ->
            # [B,K,M,cD,L]
            ys = ys.reshape(
                B,
                K,
                M,
                cD,
                L,
            )

            # 四个扫描方向恢复到同一原图坐标
            ys = self._align_cross2d_outputs(
                ys,
                H,
                W,
            )
            # ys: [B,4,M,cD,H,W]

            # 4*M个方向-通道组节点执行图注意力
            ys = self.dg_gat(ys)

            # 通道组拼接回D通道
            # 四个方向执行逐元素融合
            y = ys.reshape(
                B,
                K,
                D,
                H,
                W,
            ).sum(dim=1)

        else:
            # =================================
            # 原始CrossMerge分支
            # =================================
            ys = ys.reshape(
                B,
                K,
                D,
                H,
                W,
            )

            if K == 4:
                y = cross_merge_fn(
                    ys,
                    in_channel_first=True,
                    out_channel_first=True,
                    scans=scan_id,
                    force_torch=scan_force_torch,
                )
            else:
                y = cross_merge_fn_k2(
                    ys,
                    in_channel_first=True,
                    out_channel_first=True,
                    scans=scan_id,
                    force_torch=scan_force_torch,
                )

            y = y.reshape(
                B,
                D,
                H,
                W,
            )

        # =====================================
        # 5. 输出格式与归一化
        # =====================================
        if not self.channel_first:
            y = (
                y.reshape(B, D, L)
                .transpose(1, 2)
                .reshape(B, H, W, D)
                .contiguous()
            )

        y = self.out_norm(y)

        return y.to(
            dtype=x.dtype
        )