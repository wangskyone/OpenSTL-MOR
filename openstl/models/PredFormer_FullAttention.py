import sys
sys.path.append("/nas_data/LSH/OpenSTL-MOR/")
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import os
from openstl.utils import measure_throughput
from fvcore.nn import FlopCountAnalysis, flop_count_table
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from openstl.modules import Attention, PreNorm, FeedForward
import math

class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class GatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)       
            
    def forward(self, x):
        for attn, ff, drop_path1,drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))
        self.ff = PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout))
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(x))
        x = x + self.drop_path2(self.ff(x))
        return x


class MoDGatedTrans(nn.Module):
    def __init__(
        self,
        dim: int = None,                   # token 的 embedding 维度（d）
        capacity_factor: int = 0.5,       # 选取 top-k 的比例因子（后面会用 s * capacity_factor 计算 top_k）
        aux_loss_on: bool = False,         # 是否启用辅助损失
        mlp_ratio=4., drop_path=0.1,
        heads=8, dim_head=32, attn_dropout=0., dropout=0.,layer_i=0,depth=6,
        **kwargs,
    ):
        super().__init__()                 # 调用父类构造
        self.dim = dim                     # 保存 embedding 维度
        self.capacity_factor = capacity_factor  # 保存 capacity 因子（用于计算 top_k）
        self.layer_i = layer_i
        self.depth=depth
        self.transformer_block = TransformerBlock(  # 初始化一个 transformer block 作为子模块
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path=drop_path
        )
        
        self.aux_loss_on = aux_loss_on     # 保存 aux loss flag

        self.router = nn.Linear(dim, 1, bias=False)   # 路由器：线性层把每个 token 的 d 维向量映射为一个标量 logit（用于 top-k 排序）

        self.aux_router = nn.Sequential(               # 备用的辅助路由器（目前类里并未使用到），是个小 MLP
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
        )


    def forward(
        self,
        x: Tensor = None,                 # 输入 x，形状通常为 [B, S, D]（batch, seq_len, dim）
        **kwargs,
    ) -> Tensor:
        b, s, d = x.shape             # 获取输入形状：b=batch，s=序列长度，d=embedding 维度
        device = x.device             # 设备（cuda/cpu），这里取得但实际代码中并未再次使用 device 变量
        
        # if self.layer_i<2:
        #     # Top k
        #     top_k = int(s)   # 计算要选取的 token 数量 top_k（例如 capacity_factor=0.25 则选 25% 的 token）
        # elif self.layer_i==self.depth-1:
        #     top_k = int(s)
        # else:
        #     top_k = int(s * self.capacity_factor)
        top_k = int(s * self.capacity_factor)

        # Scalar weights for each token
        router_logits = self.router(x)          # 对每个 token 计算路由 logit，形状 [B, S, 1]
                                                # 每个 token 有一个标量表示“重要性”或“被选概率（未归一化）”

        # Equation 1
        token_weights, token_index = torch.topk(
            router_logits, top_k, dim=1, sorted=False
        )
        # torch.topk 返回两个张量：
        # - token_weights: top_k 的 logit 值，形状 [B, top_k, 1]
        # - token_index: 对应的索引（在原序列维度上的位置），形状 [B, top_k, 1]（索引类型为 long）

        # Selected
        selected_tokens, index = torch.sort(token_index, dim=1)
        # 对 topk 索引按序列位置排序：
        # - selected_tokens: 将 token_index 中的索引值按升序排序后得到的值，形状 [B, top_k, 1]；这些是要被选中的原序列位置（0..s-1）
        # - index: 表示排序后每个元素在原 token_index 中的位置（可用于重排对应的权重），形状 [B, top_k, 1]
        # 目的：把被选位置按时间顺序（或索引顺序）排列，便于后续 transformer 按自然顺序处理

        # Select idx
        indices_expanded = selected_tokens.expand(-1, -1, self.dim)
        # selected_tokens 形状 [B, top_k, 1] -> expand 成 [B, top_k, D]
        # 这是为了配合 torch.gather 的索引格式（index 必须和要索引的张量在非 index 维度形状一致）

        # Filtered topk tokens with capacity c
        filtered_x = torch.gather(
            input=x, dim=1, index=indices_expanded
        )
        # 从原始 x([B,S,D]) 中按 indices_expanded 在 dim=1（序列维）上采样出被选的 top_k token
        # 结果 filtered_x 形状为 [B, top_k, D]
        # 注意：torch.gather 要求 index 为 long 类型且 index.shape == src.shape（除了被索引的 dim 的大小）
        # print(filtered_x.shape)  # DEBUG 打印：通常是 [B, top_k, D]

        # I think filtered_x goes through the transformer block?
        x_out = self.transformer_block(filtered_x)
        # 把被选出来的 token（按序排列）喂给 transformer_block（外部传入的模块）进行处理
        # 假设 transformer_block 对输入 [B, top_k, D] 输出 [B, top_k, D']（通常 D'==D）

        # Softmax router weights
        token_weights = F.softmax(token_weights, dim=1)
        # 对 top_k 的 router logits 做 softmax 归一化（在 top_k 维度上）
        # token_weights 形状为 [B, top_k, 1]，softmax 之后表示每个被选 token 的归一化权重

        # Selecting router weight by idx
        r_weights = torch.gather(token_weights, dim=1, index=index)
        # 这里把 token_weights 根据之前 torch.sort 得到的 index 重排：
        # 目的是让权重的顺序对齐到 selected_tokens 的顺序（即与 filtered_x 的顺序一致）
        # r_weights 形状为 [B, top_k, 1]

        # Multiply by router weights
        xw_out = r_weights * x_out
        # 用权重对 transformer 输出按 token 加权（broadcast），结果形状 [B, top_k, D]

        # Out
        out = torch.scatter_add(
            input=x, dim=1, index=indices_expanded, src=xw_out
        )
        # 将加权后的 selected token 输出放回到原始序列对应位置：
        # - scatter_add 会把 src 的每个元素累加到 input 的指定 index 位置上
        # - indices_expanded 形状 [B, top_k, D]，src xw_out 形状 [B, top_k, D]
        # - 输出 out 的形状和 input x 相同，即 [B, S, D]
        # 注：如果 index 中存在重复位置（通常不会），scatter_add 会将多个 src 值累加到同一位置

        # Aux loss
        if self.aux_loss_on is not False:
            aux_loss = self.aux_loss(
                out, router_logits, selected_tokens
            )
            return out, aux_loss
        return out, router_logits
        
    def aux_loss(
        self,
        x: Tensor,
        router_logits: Tensor,
        selected_tokens: Tensor,
    ):
        b, s, d = x.shape

        router_targets = torch.zeros_like(router_logits).view(-1)

        router_targets[selected_tokens.view(-1)] = 1.0
        aux_router_logits = self.aux_router(
            x.detach().view(b * s, -1)
        )
        return F.binary_cross_entropy_with_logits(
            aux_router_logits.view(-1), router_targets
        )

class PredFormer_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
        # print(model_config)
        self.image_height = model_config['height']
        self.image_width = model_config['width']
        self.patch_size = model_config['patch_size']
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = model_config['pre_seq']
        self.dim = model_config['dim']
        self.num_channels = model_config['num_channels']
        self.num_classes = self.num_channels
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']  # Ensure this is defined
        self.depth = model_config['depth']  # Ensure this is defined
        
        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by the patch size.'
        self.patch_dim = self.num_channels * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
            )
        
        self.pos_embedding = nn.Parameter(sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
                                               requires_grad=False).view(1, self.num_frames_in, self.num_patches, self.dim)

        self.space_temporal_transformer = GatedTransformer(self.dim, self.Ndepth, self.heads, self.dim_head, 
                                                self.dim * self.scale_dim, self.dropout, self.attn_dropout, self.drop_path)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_channels * self.patch_size ** 2)
            ) 
                

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Patch Embedding
        x = self.to_patch_embedding(x)
        
        # Posion Embedding
        x += self.pos_embedding.to(x.device)
        
        b, t, n, _ = x.shape        
        
        # PredFormer Encoder
        x = rearrange(x, 'b t n d -> b (t n) d')
        x = self.space_temporal_transformer(x)      
        
        # MLP head        
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        
        return x


class Mod_PredFormer_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
        # print(model_config)
        self.image_height = model_config['height']
        self.image_width = model_config['width']
        self.patch_size = model_config['patch_size']
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = model_config['pre_seq']
        self.dim = model_config['dim']
        self.num_channels = model_config['num_channels']
        self.num_classes = self.num_channels
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']  # Ensure this is defined
        self.depth = model_config['depth']  # Ensure this is defined
        self.capacity_factor = model_config['capacity_factor']  # Ensure this is defined

        
        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by the patch size.'
        self.patch_dim = self.num_channels * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
            )
        
        self.pos_embedding = nn.Parameter(sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
                                               requires_grad=False).view(1, self.num_frames_in, self.num_patches, self.dim)

        self.space_temporal_transformer = nn.ModuleList([])
        for i in range(self.Ndepth):
            self.space_temporal_transformer.append(
                MoDGatedTrans(self.dim, capacity_factor=self.capacity_factor, aux_loss_on=False, mlp_ratio=self.scale_dim, 
                              drop_path=self.drop_path, heads=self.heads, dim_head=self.dim_head, 
                              attn_dropout=self.attn_dropout, dropout=self.dropout, layer_i=i, depth=self.Ndepth)
            )

        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_channels * self.patch_size ** 2)
            ) 
                

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Patch Embedding
        x = self.to_patch_embedding(x)
        
        # Posion Embedding
        x += self.pos_embedding.to(x.device)
        
        b, t, n, _ = x.shape        
        router_logits = []
        # PredFormer Encoder
        x = rearrange(x, 'b t n d -> b (t n) d')
        # x = self.space_temporal_transformer(x)      
        for layer in self.space_temporal_transformer:
            x, logits = layer(x)   
            router_logits.append(logits)
        
        # MLP head        
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        
        return x, router_logits

class Conv1dSEBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, reduction=4, dropout=0.1):
        """
        Conv1d + Squeeze-Excitation block
        Args:
            in_dim: 输入维度 (如 T*D)
            out_dim: 输出维度 (通常为 D)
            kernel_size: Conv1d 核大小 (推荐 3)
            reduction: SE 降维比
            dropout: Dropout 比例
        """
        super().__init__()
        hidden_dim = max(in_dim // 2, out_dim)

        # ① 特征提取：Conv1d 建模时序-通道相关性
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim,
                      kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ② SE 通道注意力（全连接实现）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                   # 全局聚合
            nn.Conv1d(hidden_dim, hidden_dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // reduction, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # ③ 输出映射：降维 / 融合
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=out_dim,
                      kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim)
        )

        # 可选残差
        self.use_residual = (in_dim == out_dim)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape

        # Conv1d 期望输入 [B, C, N]
        x = x.transpose(1, 2)          # → [B, C, N]
        out = self.conv(x)             # → [B, hidden_dim, N]

        # SE 模块：加权
        w = self.se(out)               # → [B, hidden_dim, 1]
        out = out * w                  # 通道加权

        # 输出投影
        out = self.proj(out)           # → [B, out_dim, N]
        out = out.transpose(1, 2)      # 回到 [B, N, out_dim]

        # 残差连接（如果维度匹配）
        if self.use_residual and out.shape == x.transpose(1, 2).shape:
            out = out + x.transpose(1, 2)

        return out

class Patch_Predformer_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
        # print(model_config)
        self.image_height = model_config['height']
        self.image_width = model_config['width']
        self.patch_size = model_config['patch_size']
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = model_config['pre_seq']
        self.dim = model_config['dim']
        self.num_channels = model_config['num_channels']
        self.num_classes = self.num_channels
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']  # Ensure this is defined
        self.depth = model_config['depth']  # Ensure this is defined
        self.capacity_factor = model_config['capacity_factor']  # Ensure this is defined

        
        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by the patch size.'
        self.patch_dim = self.num_channels * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
            )
        
        self.pos_embedding = nn.Parameter(sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
                                               requires_grad=False).view(1, self.num_frames_in, self.num_patches, self.dim)
        
        self.conv1d_se_block = Conv1dSEBlock(in_dim=self.num_frames_in * self.dim, out_dim=self.dim, kernel_size=3, reduction=4, dropout=0.1)   

        self.space_temporal_transformer = nn.ModuleList([])
        for i in range(self.Ndepth):
            self.space_temporal_transformer.append(
                MoDGatedTrans(self.dim, capacity_factor=self.capacity_factor, aux_loss_on=False, mlp_ratio=self.scale_dim, 
                              drop_path=self.drop_path, heads=self.heads, dim_head=self.dim_head, 
                              attn_dropout=self.attn_dropout, dropout=self.dropout, layer_i=i, depth=self.Ndepth)
            )

        self.temp_mlp = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim * self.num_frames_in)
            ) 
                
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_channels * self.patch_size ** 2)
            ) 
                

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Patch Embedding
        x = self.to_patch_embedding(x)
        
        # Posion Embedding
        x += self.pos_embedding.to(x.device)
        
        b, t, n, _ = x.shape        
        x = x.permute(0, 2, 1, 3).reshape(b,n,-1)  # B, N, T*D

        x = self.conv1d_se_block(x)           # [B, N, D]
        # PredFormer Encoder
        # x = rearrange(x, 'b t n d -> b (t n) d')
        # x = self.space_temporal_transformer(x)      
        for layer in self.space_temporal_transformer:
            x = layer(x)   
        x = self.temp_mlp(x).view(b,n,t,self.dim).permute(0,2,1,3)  # B, T, N, D
        x = rearrange(x, 'b t n d -> b (t n) d')
        # MLP head        
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        
        return x

# model_config = {
#     # image h w c
#     'height': 128,
#     'width': 128,
#     'num_channels': 1,
#     # video length in and out
#     'pre_seq': 10,
#     'after_seq': 20,
#     # patch size
#     'patch_size': 8,
#     'dim': 256, 
#     'heads': 8,
#     'dim_head': 32,
#     # dropout
#     'dropout': 0.1,
#     'attn_dropout': 0.1,
#     'drop_path': 0.1,
#     'scale_dim': 4,
#     # depth
#     'depth': 1,
#     'Ndepth': 12,
#     'capacity_factor': 1,
# }

# model = Patch_Predformer_Model(model_config)
# x = torch.rand(1, 10, 1, 128, 128)
# output = model(x)
# print(output.shape)  # [B, T, C, H, W]
# # # Calculate FLOPs
# flops = FlopCountAnalysis(model, x)
# print(f'Number of flops: {flop_count_table(flops)}')