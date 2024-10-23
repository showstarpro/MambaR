# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import Mlp, DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights


import math

from collections import namedtuple

from utils import get_cls_idx, get_cls_idx_star
from mamba_custom import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from pos_embed import *

__all__ = [
    'mambar_tiny_patch16_224', 'mambar_small_patch16_224',
    'mambar_base_patch16_224', 'mambar_large_patch16_224'
    ]

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, mask):
        B, N, C = q.shape
        q = self.q(q).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn += mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self. attn2 = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_1 = norm_layer(dim)
        self.norm2_2 = norm_layer(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, kv, mask):
        q = q + self.attn2(self.norm2_1(q), self.norm2_2(kv), mask)
        q = q + self.mlp(self.norm2(q))
        return q


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // patch_size[0] + 1, (img_size[1] - patch_size[1]) // patch_size[1] + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba,
                        layer_idx=layer_idx,
                        biscan=True,
                        **ssm_cfg,
                        **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    block = Block(d_model, mixer_cls,
                      norm_cls=norm_cls,
                      drop_path=drop_path,
                      fused_add_norm=fused_add_norm,
                      residual_in_fp32=residual_in_fp32,)
   
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class STAR(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16,
                 depth=24, 
                 embed_dim=192,
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 num_cls_tokens=1,
                 cls_reduce=1,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.skip = [12, 16, 20, 24]

        self.num_cls_tokens = num_cls_tokens
        self.cls_reduce = cls_reduce

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=channels,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if self.num_cls_tokens > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, num_cls_tokens, self.embed_dim))
            self.pos_embed_cls = nn.Parameter(
                torch.zeros(1, num_cls_tokens, self.embed_dim), requires_grad=False)
            H, W = self.patch_embed.grid_size
            # self.token_idx, self.cls_positions = get_cls_idx(H, W, num_cls_tokens, cross=True)
            self.token_idx, self.cls_positions = get_cls_idx_star(H, W, num_cls_tokens)
            
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        if cls_reduce > 1:
            self.neck = nn.Linear(self.num_features, self.num_features // cls_reduce, bias=False)
            self.norm_neck = (nn.LayerNorm if not rms_norm else RMSNorm)(
                embed_dim * num_cls_tokens // cls_reduce, eps=norm_epsilon, **factory_kwargs)
            
        if num_classes < 1:
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(self.num_features * (num_cls_tokens // cls_reduce), num_classes)

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        #----------------------------------
        # STAR Decoder
        self.dec_embed_dim = 512
        self.ar_token = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.dec_embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.enc2dec = nn.Linear(embed_dim * 4, self.dec_embed_dim * 4)
        self.dec_block = nn.ModuleList([
            DecoderBlock(self.dec_embed_dim, self.dec_embed_dim // 64, 4, qkv_bias=True, qk_scale=None,
                         norm_layer=nn.LayerNorm)
            for i in range(4)])
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.norm_3 = nn.LayerNorm(embed_dim)
        self.norm_4 = nn.LayerNorm(embed_dim)
        self.ar_norm = nn.LayerNorm(self.dec_embed_dim)
        self.ar_pred = nn.Linear(self.dec_embed_dim, 768)

        self.dec_pos_embed_cls = nn.Parameter(
                torch.zeros(1, num_cls_tokens, self.dec_embed_dim), requires_grad=False)

        #----------------------------------


        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)


        #----------------------------
        # pos_embed init
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        dec_pos_embed = get_2d_sincos_pos_embed(self.dec_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                                cls_token=False)
        self.dec_pos_embed.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0))
        trunc_normal_(self.ar_token, std=.02)

        pos_embed_cls = get_1d_sincos_pos_embed(self.pos_embed_cls.shape[-1], int(self.num_cls_tokens), cls_token=False)
        self.pos_embed_cls.data.copy_(torch.from_numpy(pos_embed_cls).float().unsqueeze(0))
        dec_pos_embed_cls = get_1d_sincos_pos_embed(self.dec_pos_embed_cls.shape[-1], int(self.num_cls_tokens),
                                                cls_token=False)
        self.dec_pos_embed_cls.data.copy_(torch.from_numpy(dec_pos_embed_cls).float().unsqueeze(0))

        #----------------------------

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.dec_block.apply(self.atten_init_weights)
        self.register_buffer("mask", self.mask_generate(14, 15))
    def mask_generate(self, segment, tokens_per_segment):
        mask = torch.tril(torch.ones((segment, segment), dtype=torch.float))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
        mask = torch.repeat_interleave(mask, repeats=tokens_per_segment, dim=0)
        mask = torch.repeat_interleave(mask, repeats=tokens_per_segment, dim=1)
        return mask
    def atten_init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "pos_embed_cls"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token       
        x = self.patch_embed(x)
        B, _, _ = x.shape

        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.num_cls_tokens > 0:
            cls_token = self.cls_token.expand(B, -1, -1) + self.pos_embed_cls
            x = torch.cat([x, cls_token], dim=1)[:, self.token_idx]

        # mamba impl
        residual = None
        hidden_states = x
        features = []
        count = 0

        for n, layer in enumerate(self.layers):          
            hidden_states, residual = layer(
                hidden_states, residual,
                inference_params=inference_params)
            count += 1
            if count in self.skip:
                features.append(hidden_states)
        features = [self.norm_1(features[0]), self.norm_2(features[1]),
                    self.norm_3(features[2]),self.norm_4(features[3])]
        features = self.enc2dec(torch.cat(features, dim=-1))     

        B, N, C = features.shape
        assert N==14*15
        return features.reshape(B, N, C//4, 4)

        
        # if not self.fused_add_norm:
        #     if residual is None:
        #         residual = hidden_states
        #     else:
        #         residual = residual + self.drop_path(hidden_states)
        #     hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        # else:
        #     # Set prenorm=False here since we don't need the residual
        #     fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        #     hidden_states = fused_add_norm_fn(
        #         self.drop_path(hidden_states),
        #         self.norm_f.weight,
        #         self.norm_f.bias,
        #         eps=self.norm_f.eps,
        #         residual=residual,
        #         prenorm=False,
        #         residual_in_fp32=self.residual_in_fp32,
        #     )
       
        # if self.cls_reduce > 1:
        #     hidden_states = self.neck(hidden_states[:, self.cls_positions]).view(B, -1)
        #     return self.norm_neck(hidden_states)

        # if self.num_cls_tokens > 0:
        #     return hidden_states[:, self.cls_positions].view(B, -1)

        # return hidden_states
    
    def forward_decoder(self, latent_ar):
        # embed tokens
        B, N, C, depth = latent_ar.shape

        ar_token = self.ar_token + self.dec_pos_embed
        reg_token = self.ar_token + self.dec_pos_embed_cls
        
        ar_token = torch.cat([ar_token, reg_token], dim=1)[:, self.token_idx]
        ar_token = ar_token.repeat(B, 1, 1)

        # apply Transformer blocks
        count = 0
        for blk in self.dec_block:
            ar_token = blk(ar_token, latent_ar[:, :, :, count], self.mask)
            count += 1
        ar_token = self.ar_norm(ar_token)
        ar_token = self.ar_pred(ar_token)
        return ar_token

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def forward_loss(self, imgs, pred):
        target = self.patchify(imgs)
        if True:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        B, N, C = target.shape
        cls_token = self.cls_token.expand(B, -1, -1)
        target = torch.cat([target, cls_token], dim=1)[:, self.token_idx]

        loss = (pred - target) ** 2
        return loss



    def forward(self, x, return_features=False, inference_params=None):
        labels = x.clone()
        features = self.forward_features(x, inference_params)
        pred = self.forward_decoder(features)
        loss = self.forward_loss(labels, pred)
        return loss.mean(-1).mean()
        # if return_features:
        #     return x
        # return self.head(x)

@register_model
def star_tiny_patch16_224(pretrained=False, **kwargs):
    model = STAR(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, num_cls_tokens=12, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def star_small_patch16_224(pretrained=False, **kwargs):
    model = STAR(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, num_cls_tokens=14, cls_reduce=1, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_small_patch16_224.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def star_base_patch16_224(pretrained=False, **kwargs):
    model = STAR(
        patch_size=16, embed_dim=768, depth=24, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, num_cls_tokens=14, cls_reduce=1, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_base_patch16_224.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def star_large_patch16_224(pretrained=False, **kwargs):
    model = STAR(
        patch_size=16, embed_dim=1024, depth=48, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, num_cls_tokens=16, cls_reduce=8, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_large_patch16_224.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model