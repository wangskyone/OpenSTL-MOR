import sys
sys.path.append('/nas_data/LSH/OpenSTL-MOR')
import torch
import torch.nn as nn
from timm.models.swin_transformer import PatchEmbed, PatchMerging
from timm.models.vision_transformer import Block as vitblock
from timm.layers import to_2tuple


class LinearRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, out_dim, bias=False)
        self.router.weight.data.normal_(mean=0.0, std=config.initializer_range)
        
    def forward(self, x):
        return self.router(x)
    
    
class MLPRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, out_dim, bias=False)
        )
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=config.initializer_range)
    
    def forward(self, x):
        return self.router(x)
    
    
class WideMLPRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, out_dim, bias=False)
        )
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=config.initializer_range)
    
    def forward(self, x):
        return self.router(x)


class ViTLSTMCell(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, depth,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, flag=None):
        """
        Args:
        flag:  0 UpSample   1 DownSample  2 STconvert
        """
        super(ViTLSTMCell, self).__init__()

        self.STBs = nn.ModuleList(
            ViTB(i, dim=dim, input_resolution=input_resolution, depth=depth, 
                num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path, norm_layer=norm_layer, flag=flag)
            for i in range(depth))

    def forward(self, xt, hidden_states):
        """
        Args:
        xt: input for t period 
        hidden_states: [hx, cx] hidden_states for t-1 period
        """
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C).to(xt.device)
            cx = torch.zeros(B, L, C).to(xt.device)

        else:
            hx, cx = hidden_states
        
        outputs = []
        for index, layer in enumerate(self.STBs):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)
            else:
                if index % 2 == 0:
                    x = layer(outputs[-1], xt)
                    outputs.append(x)
                if index % 2 == 1:
                    x = layer(outputs[-1], None)
                    outputs.append(x)
                
        o_t = outputs[-1]
        Ft = torch.sigmoid(o_t)

        cell = torch.tanh(o_t)

        Ct = Ft * (cx + cell)
        Ht = Ft * torch.tanh(Ct)

        return Ht, (Ht, Ct)


class ViTB(vitblock):
    def __init__(self, index, dim, input_resolution, depth, num_heads, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm, flag=None,**args):
        if flag == 0:
            drop_path = drop_path[depth - index - 1]
        elif flag == 1:
            drop_path = drop_path[index]
        elif flag == 2:
            drop_path = drop_path
        super(ViTB, self).__init__(dim=dim, num_heads=num_heads, 
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                  drop_path=drop_path,
                                  norm_layer=norm_layer)
        self.input_resolution = input_resolution
        self.red = nn.Linear(2 * dim, dim)

    def forward(self, x, hx=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        if hx is not None:
            hx = self.norm1(hx)
            x = torch.cat((x, hx), -1)
            x = self.red(x)
        # x = x.view(B, H, W, C)

        x = shortcut + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        return x

        
class PatchInflated(nn.Module):
    r""" Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()

        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution

        self.Conv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3),
                                       stride=stride, padding=padding, output_padding=output_padding)
        self.Conv2 = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(3, 3),
                                       stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.Conv(x)
        x = self.Conv2(x)

        return x
       
class PatchExpanding(nn.Module):
    r""" Patch Expanding Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = x.reshape(B, H, W, 2, 2, C // 4)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * 2, W * 2, C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x

class STconvert(nn.Module): 
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths, num_heads, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, flag=2):
        super(STconvert, self).__init__()
        
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                      in_chans=in_chans, embed_dim=embed_dim, 
                                      norm_layer=norm_layer)
        patches_resolution = self.patch_embed.grid_size
        num_patches = patches_resolution[0] * patches_resolution[1]
         # 学习型绝对位置编码（无 cls）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.patch_inflated = PatchInflated(in_chans=in_chans, embed_dim=embed_dim,
                                            input_resolution=patches_resolution)

        self.layer = ViTLSTMCell(dim=embed_dim, 
                                  input_resolution=(patches_resolution[0], patches_resolution[1]), 
                                  depth=depths, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=drop_path_rate, norm_layer=norm_layer,
                                  flag=flag)
    def forward(self, x, h=None):

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x, hidden_state = self.layer(x, h)

        x = torch.sigmoid(self.patch_inflated(x))
        
        return x, hidden_state