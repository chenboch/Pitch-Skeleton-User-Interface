#!/usr/bin/python
# -*- coding:utf8 -*-
import math
import os

import numpy as np
import torch
import torch.nn as nn
import logging

from easydict import EasyDict
from einops import rearrange
from ..base import BaseModel

from DSTA_main.utils.common import TRAIN_PHASE
from DSTA_main.utils.utils_registry import MODEL_REGISTRY
from timm.models.layers import DropPath
from functools import partial
import torch.distributions as distributions
# from mmpose.models.heads.rle_regression_head import nets, nett, RealNVP

from mmpose.evaluation.functional import keypoint_pck_accuracy
from ..backbones.Resnet import ResNet
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model

    `Density estimation using Real NVP
    arXiv: <https://arxiv.org/abs/1605.08803>`_.

    Code is modified from `the official implementation of RLE
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    See also `real-nvp-pytorch
    <https://github.com/senya-ashukha/real-nvp-pytorch>`_.
    """

    @staticmethod
    def get_scale_net():
        """Get the scale model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def get_trans_net():
        """Get the translation model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super(RealNVP, self).__init__()

        self.register_buffer('loc', torch.zeros(2))
        self.register_buffer('cov', torch.eye(2))
        self.register_buffer(
            'mask', torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList(
            [self.get_scale_net() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList(
            [self.get_trans_net() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialization model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping form the data space to the latent space and calculate
        the log determinant of the Jacobian matrix."""

        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])  # torch.exp(s): betas
            t = self.t[i](z_) * (1 - self.mask[i])  # gammas
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""

        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det

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



from mmpose.models.utils.transformer import ScaleNorm


def rope(x, dim):
    """Applies Rotary Position Embedding to input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int | list[int]): The spatial dimension(s) to apply
            rotary position embedding.

    Returns:
        torch.Tensor: The tensor after applying rotary position
            embedding.

    Reference:
        `RoFormer: Enhanced Transformer with Rotary
        Position Embedding <https://arxiv.org/abs/2104.09864>`_
    """
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]

    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i

    position = torch.reshape(
        torch.arange(total_len, dtype=torch.int, device=x.device),
        spatial_shape)

    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = torch.unsqueeze(position, dim=-1)

    half_size = shape[-1] // 2
    freq_seq = -torch.arange(
        half_size, dtype=torch.int, device=x.device) / float(half_size)
    inv_freq = 10000**-freq_seq

    sinusoid = position[..., None] * inv_freq[None, None, :]

    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    x1, x2 = torch.chunk(x, 2, dim=-1)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Scale(nn.Module):
    """Scale vector by element multiplications.

    Args:
        dim (int): The dimension of the scale vector.
        init_value (float, optional): The initial value of the scale vector.
            Defaults to 1.0.
        trainable (bool, optional): Whether the scale vector is trainable.
            Defaults to True.
    """

    def __init__(self, dim, init_value=1., trainable=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        """Forward function."""

        return x * self.scale


class GAUBlock(nn.Module):
    """Gated Attention Unit (GAU) in RTMBlock.

    Args:
        num_token (int): The number of tokens.
        in_token_dims (int): The input token dimension.
        out_token_dims (int): The output token dimension.
        expansion_factor (int, optional): The expansion factor of the
            intermediate token dimension. Defaults to 2.
        s (int, optional): The self-attention feature dimension.
            Defaults to 128.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
        drop_path (float, optional): The drop path rate. Defaults to 0.0.
        attn_type (str, optional): Type of attention which should be one of
            the following options:

            - 'self-attn': Self-attention.
            - 'cross-attn': Cross-attention.

            Defaults to 'self-attn'.
        act_fn (str, optional): The activation function which should be one
            of the following options:

            - 'ReLU': ReLU activation.
            - 'SiLU': SiLU activation.

            Defaults to 'SiLU'.
        bias (bool, optional): Whether to use bias in linear layers.
            Defaults to False.
        use_rel_bias (bool, optional): Whether to use relative bias.
            Defaults to True.
        pos_enc (bool, optional): Whether to use rotary position
            embedding. Defaults to False.

    Reference:
        `Transformer Quality in Linear Time
        <https://arxiv.org/abs/2202.10447>`_
    """

    def __init__(self,
                 num_token=15,
                 in_token_dims=32,
                 out_token_dims=32,
                 expansion_factor=2,
                 s=16,
                 eps=1e-5,
                 dropout_rate=0.,
                 drop_path=0.,
                 attn_type='self-attn',
                 act_fn='SiLU',
                 bias=False,
                 use_rel_bias=False,
                 pos_enc=False):

        super(GAUBlock, self).__init__()
        self.s = s
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.attn_type = attn_type
        self.pos_enc = pos_enc
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.e = int(in_token_dims * expansion_factor)
 
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)

        if attn_type == 'self-attn':
            self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            self.uv = nn.Linear(in_token_dims, self.e + self.s, bias=bias)
            self.k_fc = nn.Linear(in_token_dims, self.s, bias=bias)
            self.v_fc = nn.Linear(in_token_dims, self.e, bias=bias)
            nn.init.xavier_uniform_(self.k_fc.weight)
            nn.init.xavier_uniform_(self.v_fc.weight)

        self.ln = ScaleNorm(in_token_dims, eps=eps)

        nn.init.xavier_uniform_(self.uv.weight)

        if act_fn == 'SiLU' or act_fn == nn.SiLU:
            assert digit_version(TORCH_VERSION) >= digit_version('1.7.0'), \
                'SiLU activation requires PyTorch version >= 1.7'

            self.act_fn = nn.SiLU(True)
        elif act_fn == 'ReLU' or act_fn == nn.ReLU:
            self.act_fn = nn.ReLU(True)
        else:
            raise NotImplementedError

        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = Scale(in_token_dims)
        else:
            self.shortcut = False

        self.sqrt_s = math.sqrt(s)

        self.dropout_rate = dropout_rate

        if dropout_rate > 0.:
            self.dropout = nn.Dropout(dropout_rate)

    def rel_pos_bias(self, seq_len, k_len=None):
        """Add relative position bias."""

        if self.attn_type == 'self-attn':
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            a = rope(self.a.repeat(seq_len, 1), dim=0)
            b = rope(self.b.repeat(k_len, 1), dim=0)
            t = torch.bmm(a, b.permute(0, 2, 1))
        return t

    def _forward(self, inputs):
        """GAU Forward function."""

        if self.attn_type == 'self-attn':
            x = inputs
        else:
            x, k, v = inputs

        x = self.ln(x)

        # [B, K, in_token_dims] -> [B, K, e + e + s]
        uv = self.uv(x)
        uv = self.act_fn(uv)

        if self.attn_type == 'self-attn':
            # [B, K, e + e + s] -> [B, K, e], [B, K, e], [B, K, s]
            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
            # [B, K, 1, s] * [1, 1, 2, s] + [2, s] -> [B, K, 2, s]
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta

            if self.pos_enc:
                base = rope(base, dim=1)
            # [B, K, 2, s] -> [B, K, s], [B, K, s]
            q, k = torch.unbind(base, dim=2)

        else:
            # [B, K, e + s] -> [B, K, e], [B, K, s]
            u, q = torch.split(uv, [self.e, self.s], dim=2)

            k = self.k_fc(k)  # -> [B, K, s]
            v = self.v_fc(v)  # -> [B, K, e]

            if self.pos_enc:
                q = rope(q, 1)
                k = rope(k, 1)

        # [B, K, s].permute() -> [B, s, K]
        # [B, K, s] x [B, s, K] -> [B, K, K]
        qk = torch.bmm(q, k.permute(0, 2, 1))

        if self.use_rel_bias:
            if self.attn_type == 'self-attn':
                bias = self.rel_pos_bias(q.size(1))
            else:
                bias = self.rel_pos_bias(q.size(1), k.size(1))
            qk += bias[:, :q.size(1), :k.size(1)]
        # [B, K, K]
        kernel = torch.square(F.relu(qk / self.sqrt_s))

        if self.dropout_rate > 0.:
            kernel = self.dropout(kernel)
        # [B, K, K] x [B, K, e] -> [B, K, e]
        x = u * torch.bmm(kernel, v)
        # [B, K, e] -> [B, K, out_token_dims]
        x = self.o(x)

        return x

    def forward(self, x):
        """Forward function."""

        if self.shortcut:
            if self.attn_type == 'cross-attn':
                res_shortcut = x[0]
            else:
                res_shortcut = x
            main_branch = self.drop_path(self._forward(x))
            return self.res_scale(res_shortcut) + main_branch
        else:
            return self.drop_path(self._forward(x))



class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y

@MODEL_REGISTRY.register()
class DSTA_STD_ResNet50_GAU(BaseModel):

    def __init__(self, cfg, phase, **kwargs):
        super(DSTA_STD_ResNet50_GAU, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']
        self.is_train = True if phase == TRAIN_PHASE else False
        self.freeze_hrnet_weights = cfg.MODEL.FREEZE_HRNET_WEIGHTS
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.preact = ResNet('resnet50')

        self.pretrained = cfg.MODEL.PRETRAINED
        self.embed_dim_ratio = 32
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(3, self.embed_dim_ratio))
        drop_path_rate = 0.1
        depth = 4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # embed_dim = self.embed_dim_ratio * 17
        self.Spatial_blocks = nn.ModuleList([
            GAUBlock(drop_path=dpr[i])
            for i in range(depth)])
        self.blocks = nn.ModuleList([
            GAUBlock(drop_path=dpr[i])
            for i in range(depth)])
        self.Spatial_norm = norm_layer(self.embed_dim_ratio)
        self.Temporal_norm = norm_layer(self.embed_dim_ratio)
        self.num_frame = 3
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame, self.embed_dim_ratio))
        self.weighted_mean = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        self.head_coord = nn.Sequential(
            nn.LayerNorm(self.embed_dim_ratio),
            nn.Linear(self.embed_dim_ratio, 2),
        )
        self.head_sigma = nn.Sequential(
            nn.LayerNorm(self.embed_dim_ratio),
            nn.Linear(self.embed_dim_ratio, 2),
        )
        self.flow = RealNVP()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_coord = Linear(2048, self.num_joints * self.embed_dim_ratio)
        self.joint_ordor =  [1,2,0,0,0,3,6,4,7,5,8,9,12,10,13,11,14]


    def forward(self, x, meta=None):
        batch_size = x.shape[0]
        x = torch.cat(x.split(3, dim=1), 0)
        feat = self.preact(x)
        feat = self.avg_pool(feat).reshape(batch_size * self.num_frame, -1)
        feat = self.fc_coord(feat).reshape(batch_size * self.num_frame, self.num_joints, self.embed_dim_ratio)
        feat = torch.stack(feat.split(batch_size, dim=0), dim=0)

        #SD
        feat_S = feat[-1]
        unused_keypoint_token = feat_S[:, -2:]  # shape : bs, 2, dims
        feat_S = feat_S[:, :15]
        feat_S = rearrange(feat_S, 'b (g n) c  -> b g n c', g=5)
        feat_S = feat_S + self.Spatial_pos_embed
        feat_S = rearrange(feat_S, 'b g n c -> (b g) n c')
        for blk in self.Spatial_blocks:
            feat_S = blk(feat_S)  
        feat_S = self.Spatial_norm(feat_S)
        feat_S = rearrange(feat_S, '(b g) n c  -> b g n c', g=5)
        feat_S = rearrange(feat_S, 'b g n c ->b (g n) c')
        feat_S = torch.concat([feat_S, unused_keypoint_token], dim=1)
        

        #TD
        feat_T = rearrange(feat, 'f b p c -> (b p) f c')
        feat_T = feat_T + self.Temporal_pos_embed
        for blk in self.blocks:
            feat_T = blk(feat_T)
        feat_T = self.Temporal_norm(feat_T)
        feat_T = rearrange(feat_T, '(b p) f c -> f b p c', p=self.num_joints)
        feat_T = feat_T[-1]

        #Aggregation
        feat = torch.stack((feat_T, feat_S), dim=1)
        if batch_size == 1:
            feat = self.weighted_mean(feat).squeeze().unsqueeze(0)
        else:
            feat = self.weighted_mean(feat).squeeze()
        # feat = self.weighted_mean(feat).squeeze()
        idx = self.joint_ordor
        feat = feat[:, idx]

        coord = self.head_coord(feat).reshape(batch_size, self.num_joints, 2)
        sigma = self.head_sigma(feat).reshape(batch_size, self.num_joints, 2).sigmoid()
        score = 1 - sigma
        score = torch.mean(score, dim=2, keepdim=True)
        if self.is_train:
            target = meta['target'].cuda()
            target_weight = meta['target_weight'].cuda()
            bar_mu = (coord - target) / sigma
            log_phi = self.flow.log_prob(bar_mu.reshape(-1, 2)).reshape(batch_size, self.num_joints, 1)
            nf_loss = torch.log(sigma) - log_phi

            acc = self.get_accuracy(coord, target, target_weight)
        else:
            nf_loss = None
            acc = None

        output = EasyDict(
            pred_jts=coord,
            sigma=sigma,
            maxvals=score.float(),
            nf_loss=nf_loss,
            acc=acc
        )
        return output


    # def nets(self):
    #     return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())


    # def nett():
    #     return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))


    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        N = output.shape[0]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            norm_factor=np.ones((N, 2), dtype=np.float32))
        return avg_acc

    def init_weights(self):
        logger = logging.getLogger(__name__)
        ## init_weights
        preact_name_set = set()
        for module_name, module in self.named_modules():
            # rough_pose_estimation_net 单独判断一下
            if module_name.split('.')[0] == "preact":
                preact_name_set.add(module_name)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)

                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
            else:
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
                    if name in ['weights']:
                        nn.init.normal_(module.weight, std=0.001)

        if os.path.isfile(self.pretrained):
            pretrained_state_dict = torch.load(self.pretrained)
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']
            logger.info('=> loading pretrained model {}'.format(self.pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] == '*':
                    layer_name = name.split('.')[0]
                    if layer_name in preact_name_set:
                        need_init_state_dict[name] = m
                    else:
                        new_layer_name = "preact.{}".format(layer_name)
                        if new_layer_name in preact_name_set:
                            parameter_name = "preact.{}".format(name)
                            need_init_state_dict[parameter_name] = m
            # TODO pretrained from posewarper not test
            self.load_state_dict(need_init_state_dict, strict=False)
        elif self.pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(self.pretrained))


    @classmethod
    def get_model_hyper_parameters(cls, args, cfg):
        # dilation = cfg.MODEL.DEFORMABLE_CONV.DILATION
        # dilation_str = ",".join(map(str, dilation))
        # hyper_parameters_setting = "D_{}".format(dilation_str)
        return 'Resnet50'

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = DSTA_STD_ResNet50_GAU(cfg, phase, **kwargs)
        return model
