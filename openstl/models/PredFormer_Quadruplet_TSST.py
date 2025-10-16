import sys
sys.path.append("/nas_data/LSH/OpenSTL-MOR/")
import torch
from torch import nn
from torch import Tensor
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
    
class PredFormerLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super(PredFormerLayer, self).__init__()
        
        self.ts_temporal_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                                   mlp_dim, dropout, attn_dropout, drop_path)
        self.ts_space_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                                mlp_dim, dropout, attn_dropout, drop_path)
        self.st_space_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                                mlp_dim, dropout, attn_dropout, drop_path)
        self.st_temporal_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                                   mlp_dim, dropout, attn_dropout, drop_path)

    def forward(self, x):
        b, t, n, _ = x.shape        
        x_ts, x_ori = x, x    
        
        
        # ts-t branch
        x_ts = rearrange(x_ts, 'b t n d -> b n t d')
        x_ts = rearrange(x_ts, 'b n t d -> (b n) t d')
        x_ts = self.ts_temporal_transformer(x_ts)
        
        # ts-s branch
        x_ts = rearrange(x_ts, '(b n) t d -> b n t d', b=b)
        x_ts = rearrange(x_ts, 'b n t d -> b t n d')
        x_ts = rearrange(x_ts, 'b t n d -> (b t) n d') 
        x_ts = self.ts_space_transformer(x_ts)


        # ts output branch     
        x_ts = rearrange(x_ts, '(b t) n d -> b t n d', b=b) 
  
        # add   
        # x_ts += x_ori
        
        x_st, x_ori = x_ts, x_ts     
        
        # st-s branch
        x_st = rearrange(x_st, 'b t n d -> (b t) n d')
        x_st = self.st_space_transformer(x_st)
        
        # st-t branch
        x_st = rearrange(x_st, '(b t) ... -> b t ...', b=b)  
        x_st = x_st.permute(0, 2, 1, 3) # b n T d        
        x_st = rearrange(x_st, 'b n t d -> (b n) t d')  
        x_st = self.st_temporal_transformer(x_st)

        # st output branch     
        x_st = rearrange(x_st, '(b n) t d -> b n t d', b=b)
        x_st = rearrange(x_st, 'b n t d -> b t n d', b=b) 
        
        return x_st

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')
      
class PredFormer_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
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

        self.blocks = nn.ModuleList([
            PredFormerLayer(self.dim, self.depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout, self.attn_dropout, self.drop_path)
            for i in range(self.Ndepth)
        ])

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
        
        # PredFormer Encoder
        for blk in self.blocks:
            x = blk(x)
        
        # MLP head        
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        
        return x
    
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


        # Select idx
        indices_expanded = selected_tokens.expand(-1, -1, self.dim)
        # selected_tokens 形状 [B, top_k, 1] -> expand 成 [B, top_k, D]
        # 这是为了配合 torch.gather 的索引格式（index 必须和要索引的张量在非 index 维度形状一致）

        # Filtered topk tokens with capacity c
        filtered_x = torch.gather(
            input=x, dim=1, index=indices_expanded
        )

        x_out = self.transformer_block(filtered_x)


        # Softmax router weights
        token_weights = F.softmax(token_weights, dim=1)


        # Selecting router weight by idx
        r_weights = torch.gather(token_weights, dim=1, index=index)


        # Multiply by router weights
        xw_out = r_weights * x_out


        # Out
        out = torch.scatter_add(
            input=x, dim=1, index=indices_expanded, src=xw_out
        )


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

class ModPredFormerLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1,capacity_factor=0.3,scale_dim=4):
        super(ModPredFormerLayer, self).__init__()
        
        self.ts_temporal_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                                   mlp_dim, dropout, attn_dropout, drop_path)
        
        self.ts_space_transformer = MoDGatedTrans(dim, capacity_factor=capacity_factor, aux_loss_on=False, 
                                                  mlp_ratio=scale_dim, drop_path=drop_path, heads=heads, 
                                                  dim_head=dim_head, attn_dropout=attn_dropout, dropout=dropout)
        
        self.st_space_transformer = MoDGatedTrans(dim, capacity_factor=capacity_factor, aux_loss_on=False, 
                                                  mlp_ratio=scale_dim, drop_path=drop_path, heads=heads, 
                                                  dim_head=dim_head, attn_dropout=attn_dropout, dropout=dropout)
        
        self.st_temporal_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                                   mlp_dim, dropout, attn_dropout, drop_path)

    def forward(self, x):
        b, t, n, _ = x.shape        
        x_ts, x_ori = x, x    
        
        router_logits_layers = []
        # ts-t branch
        x_ts = rearrange(x_ts, 'b t n d -> b n t d')
        x_ts = rearrange(x_ts, 'b n t d -> (b n) t d')
        x_ts = self.ts_temporal_transformer(x_ts)
        
        # ts-s branch
        x_ts = rearrange(x_ts, '(b n) t d -> b n t d', b=b)
        x_ts = rearrange(x_ts, 'b n t d -> b t n d')
        x_ts = rearrange(x_ts, 'b t n d -> (b t) n d') 
        x_ts, router_logits = self.ts_space_transformer(x_ts)
        router_logits_layers.append(router_logits)

        # ts output branch     
        x_ts = rearrange(x_ts, '(b t) n d -> b t n d', b=b) 
  
        # add   
        # x_ts += x_ori
        
        x_st, x_ori = x_ts, x_ts     
        
        # st-s branch
        x_st = rearrange(x_st, 'b t n d -> (b t) n d')
        x_st, router_logits = self.st_space_transformer(x_st)
        router_logits_layers.append(router_logits)
        # st-t branch
        x_st = rearrange(x_st, '(b t) ... -> b t ...', b=b)  
        x_st = x_st.permute(0, 2, 1, 3) # b n T d        
        x_st = rearrange(x_st, 'b n t d -> (b n) t d')  
        x_st = self.st_temporal_transformer(x_st)

        # st output branch     
        x_st = rearrange(x_st, '(b n) t d -> b n t d', b=b)
        x_st = rearrange(x_st, 'b n t d -> b t n d', b=b) 
        
        return x_st, router_logits_layers

class Mod_PredFormer_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
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

        self.blocks = nn.ModuleList([
            ModPredFormerLayer(self.dim, self.depth, self.heads, self.dim_head, 
                               self.dim * self.scale_dim, self.dropout, 
                               self.attn_dropout, self.drop_path,self.capacity_factor,self.scale_dim)
            for i in range(self.Ndepth)
        ])

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
        
        router_logits_all = []
        # PredFormer Encoder
        for blk in self.blocks:
            x, router_logits = blk(x)
            for rl in router_logits:
                router_logits_all.append(rl)
        
        # MLP head        
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        
        return x, router_logits_all

# model_config = {
#     # image h w c
#     'height': 256,
#     'width': 256,
#     'num_channels': 3,
#     # video length in and out
#     'pre_seq': 4,
#     'after_seq': 4,
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
#     'Ndepth': 3,
#     'capacity_factor': 0.3,
# }

# model = Mod_PredFormer_Model(model_config)
# x = torch.rand(1, 4, 3, 256, 256)
# output = model(x)
# if isinstance(output, tuple):
#     output = output[0]
# print(output.shape)  # [B, T, C, H, W]
# # # Calculate FLOPs
# flops = FlopCountAnalysis(model, x)
# print(f'Number of flops: {flop_count_table(flops)}')