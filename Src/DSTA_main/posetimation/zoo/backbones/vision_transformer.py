import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from functools import partial
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn2 = norm_layer(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class VisionTransformer(nn.Module):
    """ ResNet """

    def __init__(self, architecture, norm_layer=nn.BatchNorm2d):
        super(VisionTransformer, self).__init__()
        arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'in_channels': 3,
                'patch_size': 16,
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'img_size' : (256, 192),
                'interpolate_mode': 'bicubic',
                'drop_rate':0.,
                'drop_path_rate': 0.3,
                'feedforward_channels': 3072,
                'pre_norm': False,
            }),
         }
        self.arch_settings = arch_zoo[architecture]
        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = self.arch_settings['img_size']
        in_channels = self.arch_settings['in_channels']
        patch_size = self.arch_settings['patch_size']
        pre_norm = self.arch_settings['pre_norm']
        interpolate_mode = self.arch_settings['interpolate_mode']
        drop_rate = self.arch_settings['drop_rate']
        drop_path_rate = self.arch_settings['drop_path_rate']
        num_heads = self.arch_settings['num_heads']
        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=self.img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            padding = 2,
            stride=patch_size,
            bias=not pre_norm,  # disable bias if pre_norm is used(e.g., CLIP)
        )
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches ,self.embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dims, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.num_layers)])
        self.pre_norm = nn.Identity()


    def forward(self, x):
        batch_size = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        
        x = self.drop_after_pos(x)  
        x = self.pre_norm(x)
        
        for blk in self.blocks:
            x = blk(x)

        x = x.reshape(batch_size, *patch_resolution, -1)

        x = x.permute(0, 3, 1, 2)  # 假設是將 channel-last 調整為 channel-first 格式
        return x


