import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init, build_norm_layer
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         build_feedforward_network)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet.models.utils.builder import TRANSFORMER
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.utils import deprecated_api_warning, ConfigDict
import warnings
import copy
from torch.nn import ModuleList
import torch.utils.checkpoint as cp

# Disable warnings
warnings.filterwarnings("ignore")

def get_global_pos(points, pc_range):
    points = points * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
    return points

@TRANSFORMER.register_module()
class Detr3DTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 decoder=None,
                 **kwargs):
        super(Detr3DTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(self,
                query,
                query_pos,
                feat_flatten,
                spatial_flatten,
                level_start_index, 
                temp_memory, 
                temp_pos,
                attn_masks,
                reference_points, 
                pc_range, 
                data, 
                img_metas,):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        lidar2img = data['lidar2img']
        
        inter_states = self.decoder(
            query=query,
            query_pos=query_pos,
            mlvl_feats=feat_flatten,
            temp_memory=temp_memory, 
            temp_pos=temp_pos,
            reference_points=reference_points,
            spatial_flatten=spatial_flatten,
            level_start_index=level_start_index,
            pc_range=pc_range, 
            lidar2img=lidar2img, 
            img_metas=img_metas,
            attn_masks=attn_masks)

        return inter_states

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, embed_dims, *args, **kwargs):
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)

    def forward(self,
                query,
                query_pos,
                mlvl_feats,
                temp_memory, 
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range, 
                lidar2img, 
                img_metas,
                attn_masks):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        intermediate = []
        for lid, layer in enumerate(self.layers):
            
            query = layer(
                query,
                query_pos,
                mlvl_feats,
                temp_memory, 
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range, 
                lidar2img, 
                img_metas,
                attn_masks)

            intermediate.append(query)

        return torch.stack(intermediate)

# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class Detr3DTransformerDecoder(TransformerLayerSequence):
#     """Implements the decoder in DETR3D transformer.
#     Args:
#         return_intermediate (bool): Whether to return intermediate outputs.
#         coder_norm_cfg (dict): Config of last normalization layer. Default：
#             `LN`.
#     """

#     def __init__(self, *args, return_intermediate=False, **kwargs):
#         super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)
#         self.return_intermediate = return_intermediate

#     def forward(self,
#                 query,
#                 *args,
#                 reference_points=None,
#                 reg_branches=None,
#                 **kwargs):
#         """Forward function for `Detr3DTransformerDecoder`.
#         Args:
#             query (Tensor): Input query with shape
#                 `(num_query, bs, embed_dims)`.
#             reference_points (Tensor): The reference
#                 points of offset. has shape
#                 (bs, num_query, 4) when as_two_stage,
#                 otherwise has shape ((bs, num_query, 2).
#             reg_branch: (obj:`nn.ModuleList`): Used for
#                 refining the regression results. Only would
#                 be passed when with_box_refine is True,
#                 otherwise would be passed a `None`.
#         Returns:
#             Tensor: Results with shape [1, num_query, bs, embed_dims] when
#                 return_intermediate is `False`, otherwise it has shape
#                 [num_layers, num_query, bs, embed_dims].
#         """
#         output = query
#         intermediate = []
#         intermediate_reference_points = []
#         for lid, layer in enumerate(self.layers):
#             reference_points_input = reference_points
#             output = layer(
#                 output,
#                 *args,
#                 reference_points=reference_points_input,
#                 **kwargs)
#             output = output.permute(1, 0, 2)

#             if reg_branches is not None:
#                 tmp = reg_branches[lid](output)
                
#                 assert reference_points.shape[-1] == 3

#                 new_reference_points = torch.zeros_like(reference_points)
#                 new_reference_points[..., :2] = tmp[
#                     ..., :2] + inverse_sigmoid(reference_points[..., :2])
#                 new_reference_points[..., 2:3] = tmp[
#                     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                
#                 new_reference_points = new_reference_points.sigmoid()

#                 reference_points = new_reference_points.detach()

#             output = output.permute(1, 0, 2)
#             if self.return_intermediate:
#                 intermediate.append(output)
#                 intermediate_reference_points.append(reference_points)

#         if self.return_intermediate:
#             return torch.stack(intermediate), torch.stack(
#                 intermediate_reference_points)

#         return output, reference_points

@TRANSFORMER_LAYER.register_module()
class Detr3DTemporalDecoderLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 with_cp=True,
                 **kwargs):
        super().__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

        self.use_checkpoint = with_cp

    def _forward(self,
                query,
                query_pos,
                mlvl_feats,
                temp_memory, 
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range, 
                lidar2img, 
                img_metas,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                if temp_memory is not None:
                    temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
                    temp_pos = torch.cat([query_pos, temp_pos], dim=1)
                else:
                    temp_key = temp_value = query
                    temp_pos = query_pos
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=temp_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    query_pos,
                    mlvl_feats,
                    reference_points,
                    spatial_flatten,
                    level_start_index,
                    pc_range, 
                    lidar2img, 
                    img_metas,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

    def forward(self, 
                query,
                query_pos,
                mlvl_feats,
                temp_memory, 
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range, 
                lidar2img, 
                img_metas,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward, 
                query,
                query_pos,
                mlvl_feats,
                temp_memory, 
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range, 
                lidar2img, 
                img_metas,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                )
        else:
            x = self._forward(
            query,
            query_pos,
            mlvl_feats,
            temp_memory, 
            temp_pos,
            reference_points,
            spatial_flatten,
            level_start_index,
            pc_range, 
            lidar2img, 
            img_metas,
            attn_masks,
            query_key_padding_mask,
            key_padding_mask,
        )
        return x

@ATTENTION.register_module()
class Detr3DCrossAtten(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_levels*num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
      
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        
        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        
        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat

def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask

@ATTENTION.register_module()
class DeformableFeatureAggregationCuda(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_groups=8,
            num_levels=4,
            num_cams=6,
            dropout=0.1,
            num_pts=13,
            im2col_step=64,
            batch_first=True,
            bias=1.,
            ):
        super(DeformableFeatureAggregationCuda, self).__init__()
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.group_dims = (self.embed_dims // self.num_groups)
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.weights_fc = nn.Linear(self.embed_dims, self.num_groups * self.num_levels * num_pts)
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.learnable_fc = nn.Linear(self.embed_dims, num_pts * 3)
        self.cam_embed = nn.Sequential(
            nn.Linear(12, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dims),
        )
        self.drop = nn.Dropout(dropout)
        self.im2col_step = im2col_step
        self.bias = bias

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        nn.init.uniform_(self.learnable_fc.bias.data, -self.bias, self.bias)    

    def forward(self, instance_feature, query_pos,feat_flatten, reference_points, spatial_flatten, level_start_index, pc_range, lidar2img_mat, img_metas):
        bs, num_anchor = reference_points.shape[:2]
        reference_points = get_global_pos(reference_points, pc_range)
        key_points = reference_points.unsqueeze(-2) + self.learnable_fc(instance_feature).reshape(bs, num_anchor, -1, 3)

        weights = self._get_weights(instance_feature, query_pos, lidar2img_mat)

        features = self.feature_sampling(feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, img_metas)

        output = self.output_proj(features)
        output = self.drop(output) + instance_feature
        return output

    def _get_weights(self, instance_feature, anchor_embed, lidar2img_mat):
        bs, num_anchor = instance_feature.shape[:2]
        lidar2img = lidar2img_mat[..., :3, :].flatten(-2)
        cam_embed = self.cam_embed(lidar2img) # B, N, C
        feat_pos = (instance_feature + anchor_embed).unsqueeze(2) + cam_embed.unsqueeze(1)
        weights = self.weights_fc(feat_pos).reshape(bs, num_anchor, -1, self.num_groups).softmax(dim=-2)
        weights = weights.reshape(bs, num_anchor, self.num_cams, -1, self.num_groups).permute(0, 2, 1, 4, 3).contiguous()
        return weights.flatten(end_dim=1)

    def feature_sampling(self, feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, img_metas):
        bs, num_anchor, _ = key_points.shape[:3]

        pts_extand = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(lidar2img_mat[:, :, None, None], pts_extand[:, None, ..., None]).squeeze(-1)

        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        points_2d[..., 0:1] = points_2d[..., 0:1] / img_metas[0]['pad_shape'][0][1]
        points_2d[..., 1:2] = points_2d[..., 1:2] / img_metas[0]['pad_shape'][0][0]

        points_2d = points_2d.flatten(end_dim=1) #[b*6, 900, 13, 2]
        points_2d = points_2d[:, :, None, None, :, :].repeat(1, 1, self.num_groups, self.num_levels, 1, 1)

        bn, num_value, _ = feat_flatten.size()
        feat_flatten = feat_flatten.reshape(bn, num_value, self.num_groups, -1)
        # attention_weights = weights * mask
        output = MultiScaleDeformableAttnFunction.apply(
                feat_flatten, spatial_flatten, level_start_index, points_2d,
                weights, self.im2col_step)
        
        output = output.reshape(bs, self.num_cams, num_anchor, -1)

        return output.sum(1)
    
@ATTENTION.register_module()
class CrossAttentionDetr3D(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_groups=8,
            num_levels=4,
            num_cams=6,
            dropout=0.1,
            num_pts=13,
            im2col_step=64,
            batch_first=True,
            bias=1.,
            ):
        super(CrossAttentionDetr3D, self).__init__()
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.group_dims = (self.embed_dims // self.num_groups)
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.num_pts = num_pts
        
        # Cross attention layers
        self.query_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.key_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.value_proj = nn.Linear(self.embed_dims, self.embed_dims)
        
        self.weights_fc = nn.Linear(self.embed_dims, self.num_groups * self.num_levels * num_pts)
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.learnable_fc = nn.Linear(self.embed_dims, num_pts * 3)
        
        self.cam_embed = nn.Sequential(
            nn.Linear(12, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dims),
        )
        self.drop = nn.Dropout(dropout)
        self.im2col_step = im2col_step
        self.bias = bias

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        xavier_init(self.query_proj, distribution="uniform", bias=0.0)
        xavier_init(self.key_proj, distribution="uniform", bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        nn.init.uniform_(self.learnable_fc.bias.data, -self.bias, self.bias)    

    def forward(self, instance_feature, query_pos, feat_flatten, reference_points, spatial_flatten, level_start_index, pc_range, lidar2img_mat, img_metas):
        bs, num_anchor = reference_points.shape[:2]
        reference_points = get_global_pos(reference_points, pc_range)
        key_points = reference_points.unsqueeze(-2) + self.learnable_fc(instance_feature).reshape(bs, num_anchor, -1, 3)

        weights = self._get_weights(instance_feature, query_pos, lidar2img_mat)

        features = self.feature_sampling(feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, img_metas)

        # Cross attention after feature sampling
        cross_attn_feature = self.cross_attention(
            instance_feature, 
            features, 
            query_pos
        )
        
        output = self.output_proj(cross_attn_feature)
        output = self.drop(output) + instance_feature
        return output

    def cross_attention(self, query, key_value_features, query_pos):
        """
        Simplified cross attention between instance features (query) and sampled features (key/value)
        """
        bs, num_anchor = query.shape[:2]
        
        # Add positional encoding to query
        q = self.query_proj(query + query_pos)  # [bs, num_anchor, embed_dims]
        
        # Use sampled features as both key and value
        k = self.key_proj(key_value_features)  # [bs, num_anchor, embed_dims]
        v = self.value_proj(key_value_features)  # [bs, num_anchor, embed_dims]
        
        # Reshape for multi-head attention
        q = q.view(bs, num_anchor, self.num_groups, -1)
        k = k.view(bs, num_anchor, self.num_groups, -1)
        v = v.view(bs, num_anchor, self.num_groups, -1)
        
        # Compute attention scores
        attn_scores = torch.einsum('bngd,bngd->bng', q, k) / (self.group_dims ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [bs, num_anchor, num_groups]
        
        # Apply attention
        attended_values = torch.einsum('bng,bngd->bngd', attn_weights, v)
        attended_values = attended_values.reshape(bs, num_anchor, -1)
        
        return attended_values

    def _get_weights(self, instance_feature, anchor_embed, lidar2img_mat):
        bs, num_anchor = instance_feature.shape[:2]
        lidar2img = lidar2img_mat[..., :3, :].flatten(-2)
        cam_embed = self.cam_embed(lidar2img) # B, N, C
        feat_pos = (instance_feature + anchor_embed).unsqueeze(2) + cam_embed.unsqueeze(1)
        weights = self.weights_fc(feat_pos).reshape(bs, num_anchor, -1, self.num_groups).softmax(dim=-2)
        weights = weights.reshape(bs, num_anchor, self.num_cams, -1, self.num_groups).permute(0, 2, 1, 4, 3).contiguous()
        return weights.flatten(end_dim=1)

    def feature_sampling(self, feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, img_metas):
        bs, num_anchor, _ = key_points.shape[:3]

        pts_extand = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(lidar2img_mat[:, :, None, None], pts_extand[:, None, ..., None]).squeeze(-1)

        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        points_2d[..., 0:1] = points_2d[..., 0:1] / img_metas[0]['pad_shape'][0][1]
        points_2d[..., 1:2] = points_2d[..., 1:2] / img_metas[0]['pad_shape'][0][0]

        points_2d = points_2d.flatten(end_dim=1) #[b*6, 900, 13, 2]
        points_2d = points_2d[:, :, None, None, :, :].repeat(1, 1, self.num_groups, self.num_levels, 1, 1)

        bn, num_value, _ = feat_flatten.size()
        feat_flatten = feat_flatten.reshape(bn, num_value, self.num_groups, -1)
        
        output = MultiScaleDeformableAttnFunction.apply(
                feat_flatten, spatial_flatten, level_start_index, points_2d,
                weights, self.im2col_step)
        
        output = output.reshape(bs, self.num_cams, num_anchor, -1)

        return output.sum(1)