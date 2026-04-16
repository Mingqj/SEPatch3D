import math
from functools import partial

import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
from torch.utils.checkpoint import checkpoint

from projects.mmdet3d_plugin.models.utils.gpu_timer import GLOBAL_TIMER


from .eva_utils import (
    Backbone,
    PatchEmbed,
    get_abs_pos,
    window_partition,
    window_unpartition,
    VisionRotaryEmbeddingFast,
)

from projects.flexivit.flexivit_pytorch.patch_embed import FlexiPatchEmbed
from projects.flexivit.flexivit_pytorch.utils import resize_abs_pos_embed, to_2tuple



class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            qk_scale=None,
            attn_head_dim=None,
            rope=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.v_proj = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rope = rope
        self.proj = nn.Linear(all_head_dim, dim)

    def forward(self, x, *args, **kwargs):
        reorganize = False
        if len(x.shape) == 4:
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            N = H * W
            reorganize = True
        else:
            assert len(x.shape) == 3
            B, N, C = x.shape

        q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
        k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
        v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, N, C
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        ## rope
        if self.rope is not None:
            q = self.rope(q).type_as(v)
            k = self.rope(k).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1).type_as(x)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        if reorganize:
            x = x.view(B, H, W, C)

        return x


class ResBottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            bottleneck_channels,
            norm="LN",
            act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4 * 2 / 3,
            qkv_bias=True,
            drop_path=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_size=0,
            use_residual_block=False,
            rope=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            rope=rope,
        )

        from timm.models.layers import DropPath

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = SwiGLU(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            subln=True,
            norm_layer=norm_layer,
        )

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
            )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x

@BACKBONES.register_module()
class EVA_FLEXIViT(Backbone):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
            self,
            img_size=1024,
            patch_size=16,
            test_patch_size=32,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4 * 2 / 3,
            qkv_bias=True,
            drop_path_rate=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_abs_pos=True,
            use_rel_pos=False,
            rope=True,
            pt_hw_seq_len=16,
            intp_freq=True,
            window_size=0,
            global_window_size=20,
            use_checkpoint = True,
            # window_block_indexes=(),
            global_attn_indexes=(),
            residual_block_indexes=(),
            sim_fpn=None,
            use_act_checkpoint=False,
            pretrain_img_size=224,
            pretrain_use_cls_token=True,
            return_intermediate=False,
            out_feature="last_feat",
            xattn=True,
            class_token=False,
            no_embed_class=True,
            dynamic_img_size=False,
            dynamic_img_pad=False,
            pre_norm=False,
            # patch_size_seq=(8, 10, 12, 15, 16, 20, 24, 30, 40, 48),
            patch_size_seq=(16, 20, 24, 28, 32),
            base_pos_embed_size=16,
            patch_size_probs=None,
            interpolation="bicubic",
            antialias=True,
            
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        #########################################################################
        # Pre-initialize the flexi specific patch embed arguments
        embed_layer_fn = partial(
            FlexiPatchEmbed,
            patch_size_seq=patch_size_seq,
            patch_size_probs=patch_size_probs,
            grid_size=base_pos_embed_size,
            interpolation=interpolation,
            antialias=antialias,
        )

        # Position embedding resizing function
        self.resize_pos_embed = partial(
            resize_abs_pos_embed,
            old_size=base_pos_embed_size,
            interpolation=interpolation,
            antialias=antialias,
            num_prefix_tokens=1 if class_token and not no_embed_class else 0,
        )
        ##########################################################################
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.use_checkpoint = use_checkpoint

        ##########################################################################
        self.test_patch_size=test_patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.embed_dim = embed_dim
        self.no_embed_class = no_embed_class
        self.patch_size = patch_size
        ##########################################################################
        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer_fn(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            # dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        ##########################################################################

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = num_patches if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        self.half_head_dim = embed_dim // num_heads // 2
        self.hw_seq_len = img_size // patch_size
        self.pt_hw_seq_len = pt_hw_seq_len
        self.window_size = window_size
        self.intp_freq = intp_freq
        self.img_size = img_size

        self.rope_win = VisionRotaryEmbeddingFast(
            dim=self.half_head_dim,
            pt_seq_len=self.pt_hw_seq_len,
            ft_seq_len=self.window_size if self.intp_freq else None,
        )
        self.rope_glb = VisionRotaryEmbeddingFast(
            dim=self.half_head_dim,
            pt_seq_len=self.pt_hw_seq_len,
            ft_seq_len=self.hw_seq_len if self.intp_freq else None,
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                # window_size=window_size if i in window_block_indexes else global_window_size,
                window_size=window_size if i not in global_attn_indexes else global_window_size,
                use_residual_block=i in residual_block_indexes,
                # rope=self.rope_win if i in window_block_indexes else self.rope_glb,
                rope=self.rope_win if i not in global_attn_indexes else self.rope_glb,
            )
            if use_act_checkpoint:
                from fairscale.nn.checkpoint import checkpoint_wrapper

                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]
        
        self.return_intermediate = return_intermediate

        self.adapter = None
        if sim_fpn is not None:
            self.adapter = SimpleFeaturePyramid(**sim_fpn)

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    #########################################################################################
    def _pos_embed(self, x, patch_size, img_size):
        # Resize position embedding based on current patch size
        
        new_size = (
            int(img_size[0] // patch_size[0]),
            int(img_size[1] // patch_size[1]),
        )
        pos_embed = self.resize_pos_embed(self.pos_embed, new_size)
        if self.no_embed_class:
            # Position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # Position embedding has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + pos_embed
        # return self.pos_drop(x)
        return x
    #########################################################################################

    def forward(self, x, *args, **kwargs):
        #########################################################################################
        bs = x.shape[0]
        img_size = (x.shape[2], x.shape[3])
        if self.training:
            x, ps, img_size = self.patch_embed(x, None, return_image_size=True)
        else:
            x, ps, img_size = self.patch_embed(x, self.test_patch_size, return_image_size=True)
        x = self._pos_embed(x, ps, img_size)
        x = x.reshape(bs, img_size[0] // ps[0], img_size[1] // ps[1], self.embed_dim)
        #########################################################################################

        if self.return_intermediate:
            aux_outputs = list()
        GLOBAL_TIMER.event_start(f'StreamPETR-EVA-ViT/backbone')
        for i, blk in enumerate(self.blocks):
            x =  checkpoint(blk, x) if self.use_checkpoint else blk(x)
            if self.return_intermediate:
                aux_outputs.append(x.permute(0, 3, 1, 2))
            
        GLOBAL_TIMER.event_end(f'StreamPETR-EVA-ViT/backbone')

        if self.adapter is not None:
            x = x.permute(0, 3, 1, 2) # b, c, h, w 
            outputs = self.adapter(x)
            return outputs
        else:
            outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
            if self.return_intermediate:
                outputs['aux_outputs'] = aux_outputs
        return outputs


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks."):].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        scale_factors=[4, 2, 1, 0.5],
        in_channels=1024,
        out_channels=256,
        top_block=None,
        out_indices=[2, 3, 4, 5],
        norm="LN",
        square_pad=0,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors
        strides = [int(16 / scale) for scale in scale_factors]
        dim = in_channels

        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.Conv2d(dim, dim, kernel_size=2, stride=2)]
            elif scale == 0.25:
                layers = [nn.Conv2d(dim, dim, kernel_size=4, stride=4)]

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            if stage in out_indices:
                self.add_module(f"simfp_{stage}", layers)
                self.stages.append(layers)

    def forward(self, features):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        results = []
        for stage in self.stages:
            results.append(stage(features))

        return results